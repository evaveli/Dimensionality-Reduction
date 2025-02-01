import os
import numpy as np
from tabulate import tabulate
from utils import create_directory
from matplotlib import pyplot as plt


def ourpca(X, threshold=0.95, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on input data.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Input data to transform
    threshold : float, default=0.95
        Minimum cumulative explained variance ratio to determine number of components

    Returns:
    array-like of shape (n_samples, n_components)
        Transformed data in reduced dimensional space
    """
    # Create a copy of the original array
    data = np.copy(X)

    # Calculate mean of each feature (column)
    feature_means = np.mean(data, axis=0)

    # Center the data by subtracting the mean of each feature
    centered_data = data - feature_means

    # Calculate covariance matrix
    n_samples = data.shape[0]
    cov_matrix = np.dot(centered_data.T, centered_data) / (n_samples - 1)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Convert to real numbers (handling potential complex numbers)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if n_components is None:
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Find number of components needed to explain desired variance
        n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

    # Select top eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_components]

    # Transform the data
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    # Print eigenvalues and eigenvectors

    # Reconstruct the original dataset
    reconstructed_data = (
        np.dot(transformed_data, selected_eigenvectors.T) + feature_means
    )

    # Select top 3 features by variance for the reconstructed data
    feature_vars = np.var(reconstructed_data, axis=0)
    top_features_idx = np.argsort(feature_vars)[::-1][:3]

    # Plot the reconstructed dataset and save to plots/reconstructed_datasets
    output_dir = create_directory(os.path.join("plots", "reconstructed_datasets"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        reconstructed_data[:, top_features_idx[0]],
        reconstructed_data[:, top_features_idx[1]],
        reconstructed_data[:, top_features_idx[2]],
        alpha=0.6,
    )
    ax.set_xlabel(f"Feature {top_features_idx[0]}")
    ax.set_ylabel(f"Feature {top_features_idx[1]}")
    ax.set_zlabel(f"Feature {top_features_idx[2]}")
    plt.title("Reconstructed Data (Top 3 Features)")
    plt.savefig(os.path.join(output_dir, "reconstructed_data_plot.png"))
    plt.close()

    return transformed_data, eigenvalues, eigenvectors, cov_matrix, reconstructed_data


# This function will help to comapre the sckikit learn PCA and our PCA algorithms.


def compare_arrays_elementwise(array1, array2, array3, tolerance=1e-10):
    """
    Compare three numpy arrays element by element, specifically highlighting sign differences.
    """
    if not array1.shape == array2.shape == array3.shape:
        print(
            f"Arrays have different shapes: {array1.shape} vs {array2.shape} vs {array3.shape}"
        )
        return

    print(f"\nComparing arrays with shape {array1.shape}")
    print("\nChecking for differences:")
    print("-" * 100)

    total_elements = array1.size
    differences_count = 0
    sign_differences_count = 0
    value_differences_count = 0
    max_display = 10

    flat1 = array1.flatten()
    flat2 = array2.flatten()
    flat3 = array3.flatten()

    table_data = []

    headers = [
        "Index",
        "Our PCA",
        "sklearn PCA",
        "sklearn IPCA",
        "Diff 1-2 (with tolerance)",
        "Diff 2-3 (with tolerance)",
    ]

    for i in range(total_elements):
        values = [flat1[i], flat2[i], flat3[i]]
        diff_1_2 = abs(values[0] - values[1])
        diff_2_3 = abs(values[1] - values[2])
        max_abs_diff = max(diff_1_2, diff_2_3, abs(values[0] - values[2]))

        if max_abs_diff > tolerance:
            differences_count += 1
            # Check for sign flip between array1 and array2
            if abs(abs(values[0]) - abs(values[1])) < tolerance and np.sign(
                values[0]
            ) != np.sign(values[1]):
                sign_differences_count += 1
            # Check for sign flip between array2 and array3
            elif abs(abs(values[1]) - abs(values[2])) < tolerance and np.sign(
                values[1]
            ) != np.sign(values[2]):
                sign_differences_count += 1
            else:
                value_differences_count += 1

            if len(table_data) < max_display:
                table_data.append(
                    [
                        i + 1,
                        values[0],
                        values[1],
                        values[2],
                        diff_1_2 if diff_1_2 > tolerance else 0,
                        diff_2_3 if diff_2_3 > tolerance else 0,
                    ]
                )

    print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="pretty"))

    print("-" * 100)
    print(f"\nSummary:")
    print(f"Total elements compared: {total_elements}")
    print(f"Total differences found: {differences_count}")
    if differences_count > 0:
        print(f"  - Sign flips only: {sign_differences_count}")
        print(f"  - Value differences: {value_differences_count}")
        print(f"\nPercentage different: {(differences_count/total_elements)*100:.2f}%")
        if sign_differences_count == differences_count:
            print("\nNOTE: All differences are sign flips, which is expected in PCA!")
            print("This means the components are identical but in opposite directions.")
