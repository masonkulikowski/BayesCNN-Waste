import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import color as skcolor
from skimage.measure import moments_hu
import cv2
from PIL import Image
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime


class BayesFeatureExtractor:
    """
    Extract low-dimensional, interpretable features for Naive Bayes classification.

    Features extracted (52 total):
    - Color: Mean HSV (3) + Std HSV (3) = 6
    - Texture: LBP histogram (16) + Haralick (3) = 19
    - Shape: Aspect ratio (1) + Hu moments (3) = 4
    - Specular: 2
    - Metal: 1
    - Glass: 3
    - Trash: 1
    - LBP radius=2: 16
    """

    def __init__(self, config):
        self.config = config
        self.image_size = config['data']['image_size']
        self.lbp_bins = config['bayes'].get('lbp_bins', 16)
        self.lbp_radius = config['bayes'].get('lbp_radius', 1)
        self.lbp_points = config['bayes'].get('lbp_points', 8)

    @staticmethod
    def _to_numpy(image):
        return np.array(image) if isinstance(image, Image.Image) else image

    def _to_rgb_image(self, image):
        """Ensure image is RGB ndarray."""
        image_np = self._to_numpy(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        return image_np

    def _to_gray(self, image):
        """Return grayscale float image in [0, 1] or original scale."""
        image_np = self._to_numpy(image)
        if image_np.ndim == 3:
            return skcolor.rgb2gray(image_np)
        return image_np.astype(float)

    def _to_gray_uint8(self, image):
        """Return grayscale uint8 image."""
        gray = self._to_gray(image)
        return (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

    def extract_color_features(self, image):
        """
        Extract color features from HSV color space.

        Returns:
            6 features: [mean_h, mean_s, mean_v, std_h, std_s, std_v]
        """
        image_rgb = self._to_rgb_image(image)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        mean_hsv = np.mean(hsv, axis=(0, 1))
        std_hsv = np.std(hsv, axis=(0, 1))

        return np.concatenate([mean_hsv, std_hsv])

    def extract_texture_features(self, image):
        """
        Extract texture features using LBP and Haralick.

        Returns:
            19 features: LBP histogram (16) + Haralick contrast/entropy/homogeneity (3)
        """
        gray = self._to_gray(image)
        gray_uint8 = self._to_gray_uint8(image)

        lbp = local_binary_pattern(
            gray_uint8,
            P=self.lbp_points,
            R=self.lbp_radius,
            method='uniform'
        )

        lbp_hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.lbp_bins,
            range=(0, self.lbp_bins),
            density=True
        )

        glcm = graycomatrix(
            gray_uint8,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()

        haralick_features = np.array([contrast, energy, homogeneity])

        return np.concatenate([lbp_hist, haralick_features])

    def extract_shape_features(self, image):
        """
        Extract shape features.

        Returns:
            4 features: aspect_ratio (1) + first 3 Hu moments (3)
        """
        image_np = self._to_numpy(image)

        height, width = image_np.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0

        gray_uint8 = self._to_gray_uint8(image_np)

        hu = moments_hu(gray_uint8)

        hu_features = -np.sign(hu[:3]) * np.log10(np.abs(hu[:3]) + 1e-10)

        return np.concatenate([[aspect_ratio], hu_features])

    def extract_specular_features(self, image):
        """
        Extract specular reflection features (for metal detection).

        Optimized: Returns only the 2 most discriminative features.

        Returns:
            2 features: highlight_contrast, gradient_concentration
        """
        image_rgb = self._to_rgb_image(image)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2].astype(float) / 255.0
        gray = self._to_gray(image_rgb)

        bright_mask = value_channel > 0.85

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        if np.any(bright_mask):
            highlight_contrast = np.mean(grad_magnitude[bright_mask])
        else:
            highlight_contrast = 0.0

        grad_flat = grad_magnitude.flatten()
        if len(grad_flat) > 0:
            top_10_percent_threshold = np.percentile(grad_flat, 90)
            top_gradients = grad_flat[grad_flat >= top_10_percent_threshold]
            mean_all_gradients = np.mean(grad_flat)

            if mean_all_gradients > 0:
                gradient_concentration = np.mean(top_gradients) / mean_all_gradients
            else:
                gradient_concentration = 0.0
        else:
            gradient_concentration = 0.0

        return np.array([
            highlight_contrast,
            gradient_concentration
        ])

    def extract_glass_features(self, image):
        """
        Extract improved glass detection features.

        Optimized: Removed unstable Canny edge ratios, kept smooth gradient-based features.

        Returns:
            3 features: brightness_gradient_smoothness, high_freq_energy, saturation_uniformity
        """
        image_rgb = self._to_rgb_image(image)
        gray = self._to_gray(image_rgb)
        gray_uint8 = self._to_gray_uint8(image_rgb)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        h, w = gray_uint8.shape
        center = gray_uint8[h//4:3*h//4, w//4:3*w//4].mean()
        edges_brightness = np.concatenate([
            gray_uint8[0:h//4, :].ravel(),
            gray_uint8[3*h//4:, :].ravel()
        ])
        if len(edges_brightness) > 0:
            edges_mean = edges_brightness.mean()
            brightness_gradient_smoothness = abs(center - edges_mean) / 255.0
        else:
            brightness_gradient_smoothness = 0.0

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = magnitude.shape

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        radius_sq = (0.3 * min(h, w)) ** 2
        mask = ((y - cy) ** 2 + (x - cx) ** 2) >= radius_sq

        high_freq_energy = np.sum(magnitude * mask) / (np.sum(magnitude) + 1e-6)

        saturation = hsv[:, :, 1].astype(float) / 255.0
        saturation_uniformity = 1.0 / (np.std(saturation) + 0.01)

        return np.array([
            brightness_gradient_smoothness,
            high_freq_energy,
            saturation_uniformity
        ])

    def extract_trash_features(self, image):
        """
        Extract trash-specific features for heterogeneous material detection.

        Optimized: Returns only the most discriminative feature.

        Returns:
            1 feature: texture_chaos
        """
        gray_uint8 = self._to_gray_uint8(image)

        h, w = gray_uint8.shape
        quadrants = [
            gray_uint8[0:h//2, 0:w//2],
            gray_uint8[0:h//2, w//2:],
            gray_uint8[h//2:, 0:w//2],
            gray_uint8[h//2:, w//2:]
        ]

        lbp_variances = []
        for quad in quadrants:
            if quad.size > 0:
                lbp = local_binary_pattern(quad, P=8, R=1, method='uniform')
                lbp_variances.append(np.var(lbp))

        texture_chaos = np.var(lbp_variances) if len(lbp_variances) > 0 else 0.0

        return np.array([texture_chaos])

    def extract_metal_features(self, image):
        """
        Extract additional metal-specific features beyond specular.

        Optimized: Returns only the most discriminative feature.

        Returns:
            1 feature: reflection_directionality
        """
        gray_uint8 = self._to_gray_uint8(image)

        grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(grad_y, grad_x)

        hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        reflection_directionality = np.max(hist) / (np.sum(hist) + 1e-6)

        return np.array([reflection_directionality])

    def extract_spatial_asymmetry_features(self, image):
        """
        Extract spatial asymmetry features (lighting variations).

        Returns:
            2 features: lr_brightness_var_diff, tb_brightness_var_diff
        """
        image_rgb = self._to_rgb_image(image)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2].astype(float)

        h, w = value_channel.shape

        left_half = value_channel[:, :w//2]
        right_half = value_channel[:, w//2:]
        lr_brightness_var_diff = np.var(left_half) - np.var(right_half)

        top_half = value_channel[:h//2, :]
        bottom_half = value_channel[h//2:, :]
        tb_brightness_var_diff = np.var(top_half) - np.var(bottom_half)

        return np.array([lr_brightness_var_diff, tb_brightness_var_diff])

    def extract_single_scale_lbp(self, image, radius=2):
        """
        Extract single-scale LBP features (optimized version).

        Args:
            image: PIL Image or numpy array
            radius: LBP radius to use (default=2 for best performance)

        Returns:
            16 features: LBP histogram at specified radius
        """
        gray_uint8 = self._to_gray_uint8(image)

        points = 8 * radius
        lbp = local_binary_pattern(
            gray_uint8,
            P=points,
            R=radius,
            method='uniform'
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.lbp_bins,
            range=(0, self.lbp_bins),
            density=True
        )

        return hist

    def extract_features(self, image):
        """
        Extract all features from an image.

        Optimized feature set with reduced redundancy:
        - Color: 6
        - Texture (LBP + Haralick from extract_texture_features): 19
        - Shape: 4
        - Specular (optimized): 2
        - Metal (optimized): 1
        - Glass (optimized): 3
        - Trash (optimized): 1
        - Single-scale LBP radius=2: 16
        Total: 52 features (before correlation removal)

        Args:
            image: PIL Image or numpy array

        Returns:
            Feature vector (52 features)
        """
        color_feat = self.extract_color_features(image)
        texture_feat = self.extract_texture_features(image)
        shape_feat = self.extract_shape_features(image)
        specular_feat = self.extract_specular_features(image)
        metal_feat = self.extract_metal_features(image)
        glass_feat = self.extract_glass_features(image)
        trash_feat = self.extract_trash_features(image)
        lbp_feat = self.extract_single_scale_lbp(image, radius=2)

        return np.concatenate([
            color_feat,
            texture_feat,
            shape_feat,
            specular_feat,
            metal_feat,
            glass_feat,
            trash_feat,
            lbp_feat
        ])

    def augment_image(self, image):
        """
        Apply random augmentation to an image.

        Args:
            image: PIL Image

        Returns:
            Augmented PIL Image
        """
        import random
        from PIL import ImageEnhance

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=(128, 128, 128))

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        return image

    def extract_features_batch(self, dataset, desc="Extracting features", augment=False, augment_factor=1):
        """
        Extract features from a batch of images.

        Args:
            dataset: Dataset to extract from
            desc: Progress bar description
            augment: Whether to apply augmentation
            augment_factor: How many augmented versions per image (if augment=True)

        Returns:
            features, labels arrays
        """
        features_list = []
        labels_list = []

        for item in tqdm(dataset, desc=desc):
            if isinstance(item, dict):
                image = item['image']
                label = item['label']
            else:
                image, label = item

            if isinstance(image, Image.Image):
                image = image.resize((self.image_size, self.image_size))

            features = self.extract_features(image)
            features_list.append(features)
            labels_list.append(label)

            if augment:
                for _ in range(augment_factor):
                    aug_image = self.augment_image(image)
                    aug_features = self.extract_features(aug_image)
                    features_list.append(aug_features)
                    labels_list.append(label)

        return np.array(features_list), np.array(labels_list)


class BayesClassifier:
    """Naive Bayes classifier for waste classification."""

    def __init__(self, config, use_balanced_priors=False):
        """
        Initialize Bayes classifier.

        Args:
            config: Configuration dictionary
            use_balanced_priors: If True, use balanced class priors (1/n_classes for each).
                               If False, use empirical priors from training data.
        """
        self.config = config
        self.feature_extractor = BayesFeatureExtractor(config)
        self.use_balanced_priors = use_balanced_priors

        bayes_config = config.get('bayes', {})
        self.use_multinomial = bayes_config.get('use_multinomial', False)
        self.use_ovr = bayes_config.get('use_ovr', False)
        self.apply_pca = bayes_config.get('apply_pca', False)
        self.pca_components = bayes_config.get('pca_components', 15)
        self.apply_box_cox = bayes_config.get('apply_box_cox', False)
        self.remove_correlated = bayes_config.get('remove_correlated', False)
        self.correlation_threshold = bayes_config.get('correlation_threshold', 0.85)

        self.feature_mask = None
        self.kept_feature_indices = None

        if self.use_multinomial:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.pca = None
        if self.apply_pca:
            self.pca = PCA(n_components=self.pca_components)

        self.box_cox_transformer = None
        if self.apply_box_cox:
            self.box_cox_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

        if self.use_multinomial:
            base_model = MultinomialNB()
        else:
            if use_balanced_priors:
                n_classes = config['data']['num_classes']
                priors = np.ones(n_classes) / n_classes
                base_model = GaussianNB(priors=priors)
            else:
                base_model = GaussianNB()

        if self.use_ovr:
            self.model = OneVsRestClassifier(base_model)
        else:
            self.model = base_model

        self.class_names = config['data']['classes']
        self.is_fitted = False

    def fit(self, train_dataset, verbose=True, use_augmentation=None, augment_factor=2):
        """
        Train the Bayes classifier on the training dataset.

        Args:
            train_dataset: Training dataset
            verbose: Whether to print progress
            use_augmentation: Whether to use data augmentation. If None, reads from config.
            augment_factor: Number of augmented versions per image

        Returns:
            self
        """
        if use_augmentation is None:
            use_augmentation = self.config.get('augmentation', {}).get('enabled', False)

        if verbose:
            if use_augmentation:
                print(f"Extracting features with augmentation (factor={augment_factor})...")
                print(f"  Original dataset size: {len(train_dataset)}")
                print(f"  Effective size with augmentation: {len(train_dataset) * (1 + augment_factor)}")
            else:
                print("Extracting features from training data...")

        X_train, y_train = self.feature_extractor.extract_features_batch(
            train_dataset,
            desc="Training features",
            augment=use_augmentation,
            augment_factor=augment_factor
        )

        if verbose:
            print(f"Feature shape: {X_train.shape}")

        X_transformed = X_train

        if self.remove_correlated:
            if verbose:
                print(f"Removing correlated features (threshold={self.correlation_threshold})...")

            feature_variances = np.var(X_transformed, axis=0)
            zero_var_mask = feature_variances < 1e-10

            if np.any(zero_var_mask):
                zero_var_indices = np.where(zero_var_mask)[0]
                if verbose:
                    print(f"  Warning: Found {len(zero_var_indices)} zero-variance features")
                    print(f"  Removing zero-variance features before correlation analysis...")

                non_zero_var_mask = ~zero_var_mask
                X_transformed = X_transformed[:, non_zero_var_mask]

                self.feature_mask = non_zero_var_mask.copy()
            else:
                self.feature_mask = np.ones(X_transformed.shape[1], dtype=bool)

            from src.models.feature_optimizer import analyze_feature_correlations, get_redundant_features
            corr_analysis = analyze_feature_correlations(
                X_transformed,
                threshold=self.correlation_threshold
            )

            if corr_analysis['num_redundant'] > 0:
                redundant_indices = get_redundant_features(corr_analysis, self.correlation_threshold)

                temp_mask = np.ones(X_transformed.shape[1], dtype=bool)
                temp_mask[redundant_indices] = False

                X_transformed = X_transformed[:, temp_mask]

                current_kept_indices = np.where(self.feature_mask)[0]
                features_to_remove = current_kept_indices[redundant_indices]
                self.feature_mask[features_to_remove] = False

                self.kept_feature_indices = np.where(self.feature_mask)[0]

                if verbose:
                    print(f"  Removed {len(redundant_indices)} correlated features")
                    total_removed = X_train.shape[1] - X_transformed.shape[1]
                    print(f"  Features: {X_train.shape[1]} -> {X_transformed.shape[1]} (removed {total_removed} total)")
            else:
                if verbose:
                    if np.any(zero_var_mask):
                        total_removed = len(zero_var_indices)
                        print(f"  No highly correlated features found")
                        print(f"  Features: {X_train.shape[1]} -> {X_transformed.shape[1]} (removed {total_removed} zero-variance)")
                    else:
                        print("  No highly correlated features found")

                self.kept_feature_indices = np.where(self.feature_mask)[0]

        if self.apply_box_cox:
            if verbose:
                print("Applying Box-Cox transformation...")
            X_transformed = self.box_cox_transformer.fit_transform(X_transformed)

        if verbose:
            scaler_name = "MinMax scaling" if self.use_multinomial else "Standardizing"
            print(f"{scaler_name} features...")
        X_transformed = self.scaler.fit_transform(X_transformed)

        if self.apply_pca:
            if verbose:
                print(f"Applying PCA (reducing to {self.pca_components} components)...")
            X_transformed = self.pca.fit_transform(X_transformed)
            if verbose:
                variance_explained = np.sum(self.pca.explained_variance_ratio_)
                print(f"  Variance explained: {variance_explained:.2%}")

        if verbose:
            model_name = "MultinomialNB" if self.use_multinomial else "GaussianNB"
            if self.use_ovr:
                model_name = f"OVR-{model_name}"
            print(f"Training {model_name} classifier...")
            print(f"  Final feature shape: {X_transformed.shape}")
            if self.use_ovr:
                print(f"  Using One-vs-Rest ensemble ({len(self.class_names)} binary classifiers)")

        self.model.fit(X_transformed, y_train)
        self.is_fitted = True

        if verbose:
            y_train_pred = self.model.predict(X_transformed)
            train_accuracy = np.mean(y_train_pred == y_train)
            print(f"Training complete!")
            print(f"Training accuracy: {train_accuracy:.4f}")

        return self

    def _transform_features(self, X):
        """Apply all learned transformations to features."""
        X_transformed = X

        if self.feature_mask is not None:
            X_transformed = X_transformed[:, self.feature_mask]

        if self.apply_box_cox and self.box_cox_transformer is not None:
            X_transformed = self.box_cox_transformer.transform(X_transformed)

        X_transformed = self.scaler.transform(X_transformed)

        if self.apply_pca and self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)

        return X_transformed

    def _log_predictions(self, y_true, y_pred, y_proba, log_file=None):
        """
        Write detailed per-image predictions to file or stdout.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            log_file: File path to write logs (None = stdout)
        """
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_file, 'w')
        else:
            f = sys.stdout

        try:
            f.write(f"\n{'='*130}\n")
            f.write(f"DETAILED PREDICTION LOG\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*130}\n")

            header = f"{'Idx':<6} | {'True':<12} | {'Pred':<12} | {'Result':<8} |"
            for cls in self.class_names:
                header += f" {cls:<10} |"
            f.write(header + "\n")
            f.write("-" * 130 + "\n")

            for i in range(len(y_true)):
                true_class = self.class_names[y_true[i]]
                pred_class = self.class_names[y_pred[i]]
                is_correct = "ok" if y_true[i] == y_pred[i] else "fail"

                proba_str = ""
                for j, prob in enumerate(y_proba[i]):
                    proba_str += f" {prob:>9.4f} |"

                f.write(f"{i:<6} | {true_class:<12} | {pred_class:<12} | {is_correct:<8} |{proba_str}\n")

            f.write(f"{'='*130}\n")

            accuracy = np.mean(y_true == y_pred)
            f.write(f"\nSummary:\n")
            f.write(f"  Total samples: {len(y_true)}\n")
            f.write(f"  Correct: {np.sum(y_true == y_pred)} ({accuracy:.2%})\n")
            f.write(f"  Incorrect: {np.sum(y_true != y_pred)} ({1-accuracy:.2%})\n")
            f.write("\n")

            if log_file:
                print(f"Predictions logged to: {log_file}")

        finally:
            if log_file and f != sys.stdout:
                f.close()

    def predict(self, dataset):
        """
        Predict class labels for a dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            numpy array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X, _ = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features"
        )

        X_transformed = self._transform_features(X)

        return self.model.predict(X_transformed)

    def predict_proba(self, dataset):
        """
        Predict class probabilities for a dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X, _ = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features"
        )

        X_transformed = self._transform_features(X)

        return self.model.predict_proba(X_transformed)

    def evaluate(self, dataset, verbose=True, log_predictions=False, log_file=None):
        """
        Evaluate the classifier on a dataset.

        Args:
            dataset: Dataset to evaluate on
            verbose: Whether to print results
            log_predictions: If True, log detailed per-image predictions
            log_file: File path to write prediction logs (None = stdout, False = no logging)

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        X, y_true = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features for evaluation"
        )

        X_transformed = self._transform_features(X)

        y_pred = self.model.predict(X_transformed)
        y_proba = self.model.predict_proba(X_transformed)

        if log_predictions:
            self._log_predictions(y_true, y_pred, y_proba, log_file=log_file)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

        cm = confusion_matrix(y_true, y_pred)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true
        }

        if verbose:
            print(f"\n{'='*60}")
            print("Evaluation Results")
            print(f"{'='*60}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"\nPer-class metrics:")
            print(f"{'Class':<12} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Support':>8}")
            print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
            for i, class_name in enumerate(self.class_names):
                print(f"{class_name:<12} | {precision_per_class[i]:>10.4f} | "
                      f"{recall_per_class[i]:>10.4f} | {f1_per_class[i]:>10.4f} | "
                      f"{support[i]:>8}")
            print(f"{'='*60}")

        return results

    def save(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'box_cox_transformer': self.box_cox_transformer,
            'feature_mask': self.feature_mask,
            'kept_feature_indices': self.kept_feature_indices,
            'config': self.config,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            BayesClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls(model_data['config'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.pca = model_data.get('pca', None)
        classifier.box_cox_transformer = model_data.get('box_cox_transformer', None)
        classifier.feature_mask = model_data.get('feature_mask', None)
        classifier.kept_feature_indices = model_data.get('kept_feature_indices', None)
        classifier.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {filepath}")
        return classifier
