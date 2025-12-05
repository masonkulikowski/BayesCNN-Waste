import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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

    Features extracted (~26-30 total):
    - Color: Mean HSV (3) + Std HSV (3) = 6 features
    - Texture: LBP histogram (16) + Haralick (3) = 19 features
    - Shape: Aspect ratio (1) + Hu moments (3) = 4 features
    """

    def __init__(self, config):
        self.config = config
        self.image_size = config['data']['image_size']
        self.lbp_bins = config['bayes'].get('lbp_bins', 16)
        self.lbp_radius = config['bayes'].get('lbp_radius', 1)
        self.lbp_points = config['bayes'].get('lbp_points', 8)

    def extract_color_features(self, image):
        """
        Extract color features from HSV color space.

        Returns:
            6 features: [mean_h, mean_s, mean_v, std_h, std_s, std_v]
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to HSV
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Mean and std for each channel
        mean_hsv = np.mean(hsv, axis=(0, 1))
        std_hsv = np.std(hsv, axis=(0, 1))

        return np.concatenate([mean_hsv, std_hsv])

    def extract_texture_features(self, image):
        """
        Extract texture features using LBP and Haralick.

        Returns:
            19 features: LBP histogram (16) + Haralick contrast/entropy/homogeneity (3)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = skcolor.rgb2gray(image)
        else:
            gray = image

        # Convert to uint8 for processing
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        # 1. Local Binary Pattern (LBP)
        lbp = local_binary_pattern(
            gray_uint8,
            P=self.lbp_points,
            R=self.lbp_radius,
            method='uniform'
        )

        # LBP histogram
        lbp_hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.lbp_bins,
            range=(0, self.lbp_bins),
            density=True
        )

        # 2. Haralick texture features from Gray Level Co-occurrence Matrix (GLCM)
        # Compute GLCM
        glcm = graycomatrix(
            gray_uint8,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract Haralick features (averaged over all angles)
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
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 1. Aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0

        # 2. Hu moments (use first 3 for simplicity)
        # Convert to grayscale for moment calculation
        if len(image.shape) == 3:
            gray = skcolor.rgb2gray(image)
        else:
            gray = image

        # Convert to uint8
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        # Calculate Hu moments
        hu = moments_hu(gray_uint8)

        # Use first 3 Hu moments (log-transform for numerical stability)
        hu_features = -np.sign(hu[:3]) * np.log10(np.abs(hu[:3]) + 1e-10)

        return np.concatenate([[aspect_ratio], hu_features])

    def extract_specular_features(self, image):
        """
        Extract specular reflection features (for metal detection).

        Returns:
            6 features: bright_pixel_ratio, highlight_sharpness, highlight_contrast,
                       edge_sharpness, highlight_distribution, gradient_concentration
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to HSV to get Value channel
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2].astype(float) / 255.0  # Normalize to [0, 1]
        gray = skcolor.rgb2gray(image)

        # 1. Bright pixel ratio (pixels with V > 0.85)
        bright_mask = value_channel > 0.85
        bright_pixel_ratio = np.mean(bright_mask)

        # 2. Highlight sharpness (Laplacian variance in bright regions)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        if np.any(bright_mask):
            highlight_sharpness = np.var(laplacian[bright_mask])
        else:
            highlight_sharpness = 0.0

        # 3. Highlight contrast (mean gradient magnitude in bright regions)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        if np.any(bright_mask):
            highlight_contrast = np.mean(grad_magnitude[bright_mask])
        else:
            highlight_contrast = 0.0

        # 4. Edge sharpness (Laplacian variance across entire image)
        # Metals have sharp, crisp edges → high variance
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        laplacian_full = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        edge_sharpness = np.var(laplacian_full)

        # 5. Specular highlight distribution
        # Count separate bright regions and their average size
        bright_mask_uint8 = (bright_mask * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask_uint8, connectivity=8)

        # Exclude background (label 0)
        if num_labels > 1:
            # Number of separate bright regions
            num_bright_regions = num_labels - 1
            # Average size of bright regions
            region_sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
            avg_region_size = np.mean(region_sizes) if len(region_sizes) > 0 else 0.0
            # Metals have few, concentrated highlights (low count, high size)
            highlight_distribution = avg_region_size / (num_bright_regions + 1)
        else:
            highlight_distribution = 0.0

        # 6. Gradient concentration
        # Ratio of top 10% gradients to mean gradient
        # Metals have concentrated high-gradient areas
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
            bright_pixel_ratio,
            highlight_sharpness,
            highlight_contrast,
            edge_sharpness,
            highlight_distribution,
            gradient_concentration
        ])

    def extract_transparency_features(self, image):
        """
        Extract transparency features (for glass detection).

        Returns:
            3 features: interior_texture_variance, edge_entropy, saturation_edge_drop
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        gray = skcolor.rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 1. Interior texture variance (LBP variance in interior region)
        # Define interior as central 50% of image
        h, w = gray_uint8.shape
        interior = gray_uint8[h//4:3*h//4, w//4:3*w//4]

        if interior.size > 0:
            lbp_interior = local_binary_pattern(
                interior,
                P=self.lbp_points,
                R=self.lbp_radius,
                method='uniform'
            )
            interior_texture_variance = np.var(lbp_interior)
        else:
            interior_texture_variance = 0.0

        # 2. Edge entropy
        edges = cv2.Canny(gray_uint8, 50, 150)
        edge_hist, _ = np.histogram(edges.ravel(), bins=256, range=(0, 256), density=True)
        # Calculate entropy
        edge_hist = edge_hist[edge_hist > 0]  # Remove zeros
        edge_entropy = -np.sum(edge_hist * np.log2(edge_hist + 1e-10))

        # 3. Saturation edge drop (saturation difference at edges)
        saturation = hsv[:, :, 1].astype(float)
        edge_mask = edges > 0

        if np.any(edge_mask):
            edge_saturation = np.mean(saturation[edge_mask])
            non_edge_saturation = np.mean(saturation[~edge_mask])
            saturation_edge_drop = non_edge_saturation - edge_saturation
        else:
            saturation_edge_drop = 0.0

        return np.array([interior_texture_variance, edge_entropy, saturation_edge_drop])

    def extract_spatial_asymmetry_features(self, image):
        """
        Extract spatial asymmetry features (lighting variations).

        Returns:
            2 features: lr_brightness_var_diff, tb_brightness_var_diff
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value_channel = hsv[:, :, 2].astype(float)

        h, w = value_channel.shape

        # 1. Left-right brightness variance difference
        left_half = value_channel[:, :w//2]
        right_half = value_channel[:, w//2:]
        lr_brightness_var_diff = np.var(left_half) - np.var(right_half)

        # 2. Top-bottom brightness variance difference
        top_half = value_channel[:h//2, :]
        bottom_half = value_channel[h//2:, :]
        tb_brightness_var_diff = np.var(top_half) - np.var(bottom_half)

        return np.array([lr_brightness_var_diff, tb_brightness_var_diff])

    def extract_multiscale_lbp(self, image):
        """
        Extract multi-scale LBP features at radii 1, 2, 3.

        Returns:
            48 features: 16 bins × 3 radii
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = skcolor.rgb2gray(image)
        else:
            gray = image

        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        multiscale_features = []

        for radius in [1, 2, 3]:
            points = 8 * radius  # Scale points with radius
            lbp = local_binary_pattern(
                gray_uint8,
                P=points,
                R=radius,
                method='uniform'
            )

            # Histogram with 16 bins
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=self.lbp_bins,
                range=(0, self.lbp_bins),
                density=True
            )
            multiscale_features.append(hist)

        return np.concatenate(multiscale_features)

    def extract_single_scale_lbp(self, image, radius=2):
        """
        Extract single-scale LBP features (optimized version).

        Args:
            image: PIL Image or numpy array
            radius: LBP radius to use (default=2 for best performance)

        Returns:
            16 features: LBP histogram at specified radius
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = skcolor.rgb2gray(image)
        else:
            gray = image

        gray_uint8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        points = 8 * radius
        lbp = local_binary_pattern(
            gray_uint8,
            P=points,
            R=radius,
            method='uniform'
        )

        # Histogram with 16 bins
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.lbp_bins,
            range=(0, self.lbp_bins),
            density=True
        )

        return hist

    def extract_features(self, image, use_single_scale_lbp=None):
        """
        Extract all features from an image.

        Args:
            image: PIL Image or numpy array
            use_single_scale_lbp: If True, use only radius=2 LBP (53 features total).
                                 If False, use multi-scale LBP (85 features).
                                 If None, use config setting.

        Returns:
            Feature vector (53 or 85 features depending on settings)
        """
        # Check config if not specified
        if use_single_scale_lbp is None:
            use_single_scale_lbp = self.config.get('bayes', {}).get('use_single_scale_lbp', False)

        color_feat = self.extract_color_features(image)
        texture_feat = self.extract_texture_features(image)
        shape_feat = self.extract_shape_features(image)
        specular_feat = self.extract_specular_features(image)
        transparency_feat = self.extract_transparency_features(image)
        spatial_feat = self.extract_spatial_asymmetry_features(image)

        if use_single_scale_lbp:
            # Use only radius=2 LBP (16 features instead of 48)
            lbp_feat = self.extract_single_scale_lbp(image, radius=2)
            return np.concatenate([
                color_feat,        # 6
                texture_feat,      # 19
                shape_feat,        # 4
                specular_feat,     # 6 (updated with 3 new metal features)
                transparency_feat, # 3
                spatial_feat,      # 2
                lbp_feat          # 16
            ])  # Total: 56 features
        else:
            # Use multi-scale LBP (48 features)
            multiscale_lbp_feat = self.extract_multiscale_lbp(image)
            return np.concatenate([
                color_feat,           # 6
                texture_feat,         # 19
                shape_feat,           # 4
                specular_feat,        # 6 (updated with 3 new metal features)
                transparency_feat,    # 3
                spatial_feat,         # 2
                multiscale_lbp_feat   # 48
            ])  # Total: 88 features

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

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=(128, 128, 128))

        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Random brightness (0.8 to 1.2)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # Random contrast (0.8 to 1.2)
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

            # Resize to standard size
            if isinstance(image, Image.Image):
                image = image.resize((self.image_size, self.image_size))

            # Extract features from original image
            features = self.extract_features(image)
            features_list.append(features)
            labels_list.append(label)

            # Apply augmentation if requested
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

        # Get optimization settings from config
        bayes_config = config.get('bayes', {})
        self.use_multinomial = bayes_config.get('use_multinomial', False)
        self.apply_pca = bayes_config.get('apply_pca', False)
        self.pca_components = bayes_config.get('pca_components', 20)
        self.apply_box_cox = bayes_config.get('apply_box_cox', False)
        self.use_trash_rules = bayes_config.get('use_trash_rules', False)

        # Initialize appropriate scaler
        if self.use_multinomial:
            # MultinomialNB requires non-negative features
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Initialize PCA if needed
        self.pca = None
        if self.apply_pca:
            self.pca = PCA(n_components=self.pca_components)

        # Initialize Box-Cox transformer if needed
        self.box_cox_transformer = None
        if self.apply_box_cox:
            self.box_cox_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

        # Initialize model with balanced priors if requested
        if self.use_multinomial:
            # MultinomialNB doesn't support priors parameter the same way
            self.model = MultinomialNB()
        else:
            if use_balanced_priors:
                n_classes = config['data']['num_classes']
                priors = np.ones(n_classes) / n_classes
                self.model = GaussianNB(priors=priors)
            else:
                self.model = GaussianNB()

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
        # Check if augmentation is enabled
        if use_augmentation is None:
            use_augmentation = self.config.get('augmentation', {}).get('enabled', False)

        if verbose:
            if use_augmentation:
                print(f"Extracting features with augmentation (factor={augment_factor})...")
                print(f"  Original dataset size: {len(train_dataset)}")
                print(f"  Effective size with augmentation: {len(train_dataset) * (1 + augment_factor)}")
            else:
                print("Extracting features from training data...")

        # Extract features from training data
        X_train, y_train = self.feature_extractor.extract_features_batch(
            train_dataset,
            desc="Training features",
            augment=use_augmentation,
            augment_factor=augment_factor
        )

        if verbose:
            print(f"Feature shape: {X_train.shape}")

        # Apply transformations
        X_transformed = X_train

        # 1. Box-Cox transformation (if enabled)
        if self.apply_box_cox:
            if verbose:
                print("Applying Box-Cox transformation...")
            X_transformed = self.box_cox_transformer.fit_transform(X_transformed)

        # 2. Standardize/normalize features
        if verbose:
            scaler_name = "MinMax scaling" if self.use_multinomial else "Standardizing"
            print(f"{scaler_name} features...")
        X_transformed = self.scaler.fit_transform(X_transformed)

        # 3. PCA dimensionality reduction (if enabled)
        if self.apply_pca:
            if verbose:
                print(f"Applying PCA (reducing to {self.pca_components} components)...")
            X_transformed = self.pca.fit_transform(X_transformed)
            if verbose:
                variance_explained = np.sum(self.pca.explained_variance_ratio_)
                print(f"  Variance explained: {variance_explained:.2%}")

        if verbose:
            model_name = "MultinomialNB" if self.use_multinomial else "GaussianNB"
            print(f"Training {model_name} classifier...")
            print(f"  Final feature shape: {X_transformed.shape}")

        # Train Naive Bayes
        self.model.fit(X_transformed, y_train)
        self.is_fitted = True

        # Compute training accuracy
        if verbose:
            y_train_pred = self.model.predict(X_transformed)
            train_accuracy = np.mean(y_train_pred == y_train)
            print(f"Training complete!")
            print(f"Training accuracy: {train_accuracy:.4f}")

        return self

    def _transform_features(self, X):
        """Apply all learned transformations to features."""
        X_transformed = X

        # Apply transformations in same order as training
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
        # Open file or use stdout
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

            # Header
            header = f"{'Idx':<6} | {'True':<12} | {'Pred':<12} | {'Result':<8} |"
            for cls in self.class_names:
                header += f" {cls:<10} |"
            f.write(header + "\n")
            f.write("-" * 130 + "\n")

            # Per-image predictions
            for i in range(len(y_true)):
                true_class = self.class_names[y_true[i]]
                pred_class = self.class_names[y_pred[i]]
                is_correct = "✓" if y_true[i] == y_pred[i] else "✗"

                # Format probabilities
                proba_str = ""
                for j, prob in enumerate(y_proba[i]):
                    proba_str += f" {prob:>9.4f} |"

                f.write(f"{i:<6} | {true_class:<12} | {pred_class:<12} | {is_correct:<8} |{proba_str}\n")

            f.write(f"{'='*130}\n")

            # Summary statistics
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

        # Extract features
        X, _ = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features"
        )

        # Apply transformations
        X_transformed = self._transform_features(X)

        # Predict
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

        # Extract features
        X, _ = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features"
        )

        # Apply transformations
        X_transformed = self._transform_features(X)

        # Predict probabilities
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

        # Extract features and true labels
        X, y_true = self.feature_extractor.extract_features_batch(
            dataset,
            desc="Extracting features for evaluation"
        )

        # Apply transformations
        X_transformed = self._transform_features(X)

        # Predict
        y_pred = self.model.predict(X_transformed)
        y_proba = self.model.predict_proba(X_transformed)

        # Log individual predictions if requested
        if log_predictions:
            self._log_predictions(y_true, y_pred, y_proba, log_file=log_file)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

        # Confusion matrix
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
        classifier.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {filepath}")
        return classifier
