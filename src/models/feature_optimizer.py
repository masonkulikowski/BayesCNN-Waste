"""
Feature optimization utilities for Bayes classifier.
Includes correlation analysis, PCA, and feature transformations.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_feature_correlations(X, feature_names=None, threshold=0.9):
    """
    Analyze feature correlations and identify highly correlated pairs.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        threshold: Correlation threshold for flagging redundancy

    Returns:
        Dictionary with correlation matrix and redundant pairs
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated pairs (above threshold)
    redundant_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                redundant_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix[i, j],
                    'index1': i,
                    'index2': j
                })

    return {
        'corr_matrix': corr_matrix,
        'feature_names': feature_names,
        'redundant_pairs': redundant_pairs,
        'num_redundant': len(redundant_pairs)
    }


def get_redundant_features(corr_analysis, threshold=0.9):
    """
    Get list of features to remove based on correlation analysis.

    Strategy: For each highly correlated pair, keep the first one.

    Args:
        corr_analysis: Output from analyze_feature_correlations
        threshold: Correlation threshold

    Returns:
        List of feature indices to remove
    """
    to_remove = set()

    for pair in corr_analysis['redundant_pairs']:
        if abs(pair['correlation']) > threshold:
            # Keep feature1, remove feature2
            to_remove.add(pair['index2'])

    return sorted(list(to_remove))


def plot_correlation_heatmap(corr_matrix, feature_names, figsize=(12, 10)):
    """Plot correlation heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()


def test_gaussianity(X, feature_names=None, alpha=0.05):
    """
    Test which features are non-Gaussian using Shapiro-Wilk test.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    results = []

    for i, name in enumerate(feature_names):
        # Shapiro-Wilk test (null hypothesis: data is normally distributed)
        stat, p_value = stats.shapiro(X[:, i])
        is_gaussian = p_value > alpha

        results.append({
            'feature': name,
            'index': i,
            'statistic': stat,
            'p_value': p_value,
            'is_gaussian': is_gaussian
        })

    non_gaussian_indices = [r['index'] for r in results if not r['is_gaussian']]

    return {
        'results': results,
        'non_gaussian_indices': non_gaussian_indices,
        'num_non_gaussian': len(non_gaussian_indices),
        'num_gaussian': X.shape[1] - len(non_gaussian_indices)
    }


def apply_box_cox_transform(X, feature_indices=None):
    """
    Apply Box-Cox transformation to specified features.

    Args:
        X: Feature matrix
        feature_indices: Indices of features to transform (None = all)

    Returns:
        Transformed feature matrix and transformer object
    """
    X_transformed = X.copy()

    if feature_indices is None:
        feature_indices = range(X.shape[1])

    # PowerTransformer with Box-Cox (requires positive values)
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)

    # Transform only specified features
    X_transformed[:, feature_indices] = transformer.fit_transform(X[:, feature_indices])

    return X_transformed, transformer


def apply_pca(X, n_components=20, variance_threshold=0.95):
    """
    Apply PCA dimensionality reduction.

    Args:
        X: Feature matrix
        n_components: Number of components (or 'auto' for variance threshold)
        variance_threshold: If n_components='auto', keep this much variance

    Returns:
        Transformed features, PCA object, and variance info
    """
    if n_components == 'auto':
        # Find number of components to explain variance_threshold of variance
        pca_full = PCA()
        pca_full.fit(X)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    variance_info = {
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'total_variance_explained': np.sum(pca.explained_variance_ratio_)
    }

    return X_pca, pca, variance_info


def plot_pca_variance(variance_info):
    """Plot PCA variance explained."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, len(variance_info['explained_variance_ratio']) + 1),
            variance_info['explained_variance_ratio'],
            alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Variance Explained per Component')
    ax1.grid(alpha=0.3)

    # Cumulative variance
    ax2.plot(range(1, len(variance_info['cumulative_variance']) + 1),
             variance_info['cumulative_variance'],
             marker='o', linewidth=2, color='seagreen')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def remove_features(X, indices_to_remove):
    """
    Remove specified features from feature matrix.

    Args:
        X: Feature matrix
        indices_to_remove: List of column indices to remove

    Returns:
        Feature matrix with columns removed
    """
    mask = np.ones(X.shape[1], dtype=bool)
    mask[indices_to_remove] = False
    return X[:, mask]


def get_feature_names():
    """
    Get names for all 85 features.

    Returns:
        List of feature names
    """
    names = []

    # Color (6)
    names.extend(['hsv_mean_h', 'hsv_mean_s', 'hsv_mean_v',
                  'hsv_std_h', 'hsv_std_s', 'hsv_std_v'])

    # Texture - LBP single scale (16)
    names.extend([f'lbp_r1_bin{i}' for i in range(16)])

    # Haralick (3)
    names.extend(['haralick_contrast', 'haralick_energy', 'haralick_homogeneity'])

    # Shape (4)
    names.extend(['aspect_ratio', 'hu_moment_1', 'hu_moment_2', 'hu_moment_3'])

    # Specular (3)
    names.extend(['bright_pixel_ratio', 'highlight_sharpness', 'highlight_contrast'])

    # Transparency (3)
    names.extend(['interior_texture_var', 'edge_entropy', 'saturation_edge_drop'])

    # Spatial (2)
    names.extend(['lr_brightness_var_diff', 'tb_brightness_var_diff'])

    # Multi-scale LBP (48)
    names.extend([f'lbp_r1_ms_bin{i}' for i in range(16)])
    names.extend([f'lbp_r2_bin{i}' for i in range(16)])
    names.extend([f'lbp_r3_bin{i}' for i in range(16)])

    return names


def get_single_scale_lbp_indices():
    """
    Get indices to remove for single-scale LBP optimization.

    Removes radius 1 and 3, keeps only radius 2.

    Returns:
        List of 32 indices to remove
    """
    indices_to_remove = []

    # Multi-scale LBP starts at index 37 (6+16+3+4+3+3+2 = 37)
    base_idx = 37

    # Remove radius 1 (indices 37-52)
    indices_to_remove.extend(range(base_idx, base_idx + 16))

    # Keep radius 2 (indices 53-68) - skip these

    # Remove radius 3 (indices 69-84)
    indices_to_remove.extend(range(base_idx + 32, base_idx + 48))

    return indices_to_remove


def optimize_features(X, y, config):
    """
    Complete feature optimization pipeline.

    Args:
        X: Feature matrix
        y: Labels
        config: Configuration dictionary with optimization settings

    Returns:
        Optimized feature matrix and optimization info
    """
    feature_names = get_feature_names()
    optimization_steps = []

    print("Starting feature optimization pipeline...")
    print(f"Initial features: {X.shape[1]}\n")

    # Step 1: Remove multi-scale LBP if configured
    if config.get('use_single_scale_lbp', False):
        print("Step 1: Removing multi-scale LBP (keeping radius 2 only)...")
        indices_to_remove = get_single_scale_lbp_indices()
        X = remove_features(X, indices_to_remove)
        # Update feature names
        feature_names = [name for i, name in enumerate(get_feature_names())
                        if i not in indices_to_remove]
        print(f"  Removed {len(indices_to_remove)} features")
        print(f"  Current features: {X.shape[1]}\n")
        optimization_steps.append(('single_scale_lbp', len(indices_to_remove)))

    # Step 2: Correlation analysis
    if config.get('remove_correlated', False):
        print("Step 2: Analyzing feature correlations...")
        corr_threshold = config.get('correlation_threshold', 0.9)
        corr_analysis = analyze_feature_correlations(X, feature_names, corr_threshold)
        print(f"  Found {corr_analysis['num_redundant']} highly correlated pairs")

        if corr_analysis['num_redundant'] > 0:
            redundant_indices = get_redundant_features(corr_analysis, corr_threshold)
            print(f"  Removing {len(redundant_indices)} redundant features")
            X = remove_features(X, redundant_indices)
            feature_names = [name for i, name in enumerate(feature_names)
                           if i not in redundant_indices]
            print(f"  Current features: {X.shape[1]}\n")
            optimization_steps.append(('correlation', len(redundant_indices)))

    # Step 3: Box-Cox transformation
    if config.get('apply_box_cox', False):
        print("Step 3: Applying Box-Cox transformation to non-Gaussian features...")
        gaussianity = test_gaussianity(X, feature_names)
        print(f"  Non-Gaussian features: {gaussianity['num_non_gaussian']}/{X.shape[1]}")

        if gaussianity['num_non_gaussian'] > 0:
            X, transformer = apply_box_cox_transform(X, gaussianity['non_gaussian_indices'])
            print(f"  Transformed {gaussianity['num_non_gaussian']} features\n")
            optimization_steps.append(('box_cox', gaussianity['num_non_gaussian']))

    # Step 4: PCA
    if config.get('apply_pca', False):
        print("Step 4: Applying PCA dimensionality reduction...")
        n_components = config.get('pca_components', 20)
        X, pca, variance_info = apply_pca(X, n_components)
        print(f"  Reduced to {variance_info['n_components']} components")
        print(f"  Variance explained: {variance_info['total_variance_explained']:.2%}\n")
        optimization_steps.append(('pca', X.shape[1]))

    print(f"Optimization complete!")
    print(f"Final features: {X.shape[1]} (reduced by {X.shape[1] - len(get_feature_names())})")

    return X, {
        'final_shape': X.shape,
        'optimization_steps': optimization_steps,
        'feature_names': feature_names if not config.get('apply_pca', False) else None
    }
