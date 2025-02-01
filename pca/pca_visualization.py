import numpy as np
import matplotlib.pyplot as plt
from pca.pca import ourpca
import os
import pandas as pd
from cluster_algorithms.k_means import KMeans
from pca.pca import ourpca
from sklearn.cluster import OPTICS
from scipy.optimize import linear_sum_assignment


class PCAVisualizer:
    def __init__(self):
        pass

    def plot_explained_variance(self, X, dataset_name, scaling_type, title=None):
        """
        Plot cumulative explained variance ratio and individual contributions.
        """
        # Get eigenvalues from our PCA implementation
        _, eigenvalues, _, _, _ = ourpca(
            X, threshold=1.0
        )  # Using threshold=1.0 to get all components

        # Calculate variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        individual_variance = explained_variance_ratio

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot cumulative variance
        ax1.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            "bo-",
            linewidth=2,
        )
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cumulative Explained Variance Ratio")
        ax1.set_title(
            title
            or f"Cumulative Explained Variance\n{dataset_name} Dataset ({scaling_type} Scaling)"
        )
        ax1.grid(True)
        ax1.set_xticks(range(1, len(cumulative_variance) + 1))

        # Plot individual variance contributions
        ax2.bar(
            range(1, len(individual_variance) + 1),
            individual_variance,
            color="blue",
            alpha=0.6,
        )
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Individual Explained Variance Ratio")
        ax2.set_title("Individual Component Contributions")
        ax2.grid(True)
        ax2.set_xticks(range(1, len(individual_variance) + 1))

        plt.tight_layout()

        # Create directory if it doesn't exist
        os.makedirs("plots/explained_var", exist_ok=True)

        # Save plot as JPG
        save_path = (
            f"plots/explained_var/{dataset_name}_{scaling_type}_explained_variance.jpg"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300, format="jpg")
        print(f"\nPlot saved as: {save_path}")

        plt.show()

    def print_variance_analysis(self, X):
        """
        Print detailed variance analysis for different thresholds.
        """
        thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

        # Get eigenvalues from our PCA implementation
        _, eigenvalues, _, _, _ = ourpca(
            X, threshold=1.0
        )  # Using threshold=1.0 to get all components

        # Calculate cumulative variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumsum = np.cumsum(explained_variance_ratio)

        print("\nNumber of components needed for different variance thresholds:")
        print("-" * 60)
        print("Threshold    Components    Original Dimensions    Reduction (%)")
        print("-" * 60)

        for threshold in thresholds:
            n_components = np.argmax(cumsum >= threshold) + 1
            reduction = (1 - n_components / len(cumsum)) * 100
            print(
                f"{threshold:^9.2f}    {n_components:^10d}    {len(cumsum):^18d}    {reduction:^13.1f}"
            )

    def print_and_generate_latex_variance_analysis(self, X, dataset_name, scaling):
        """
        Print variance analysis and generate LaTeX table.
        """
        thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

        # Get eigenvalues from our PCA implementation
        _, eigenvalues, _, _, _ = ourpca(
            X, threshold=1.0
        )  # Using threshold=1.0 to get all components

        # Calculate cumulative variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumsum = np.cumsum(explained_variance_ratio)

        # Print analysis
        print("\nNumber of components needed for different variance thresholds:")
        print("-" * 50)
        print("Threshold    Components    Reduction (%)")
        print("-" * 50)

        # Collect data for table
        rows = []
        for threshold in thresholds:
            n_components = np.argmax(cumsum >= threshold) + 1
            reduction = (1 - n_components / len(cumsum)) * 100
            print(f"{threshold:^9.2f}    {n_components:^10d}    {reduction:^13.1f}")
            rows.append([threshold, n_components, reduction])

        # Generate LaTeX table
        table = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\begin{tabular}{|c|c|c|}\n"
            "\\hline\n"
            "Threshold & Components & Reduction (\\%) \\\\\n"
            "\\hline\n"
        )

        for row in rows:
            line = [f"{row[0]:.2f}", str(row[1]), f"{row[2]:.1f}"]
            table += " & ".join(line) + " \\\\\n\\hline\n"

        table += (
            "\\end{tabular}\n"
            f"\\caption{{PCA Variance Analysis for {scaling} {dataset_name} Dataset }}\n"
            f"\\label{{tab:pca{dataset_name}_variance}}\n"
            "\\end{table}"
        )

        # Save LaTeX code
        os.makedirs("summary/Latex_tables", exist_ok=True)
        with open(
            f"summary/Latex_tables/pca_{dataset_name}_{scaling}_variance.txt", "w"
        ) as f:
            f.write(table)

        return table

    def save_pca_projections(self, X, dataset_name, scaling_type):
        """
        Perform PCA analysis and save projected data for different thresholds.

        Parameters:
        -----------
        X : array-like
            Input data matrix
        dataset_name : str
            Name of the dataset
        scaling_type : str
            Type of scaling used
        """
        thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

        # Get eigenvalues from our PCA implementation for analysis
        _, eigenvalues, _, _, _ = ourpca(X, threshold=1.0)

        # Calculate cumulative variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumsum = np.cumsum(explained_variance_ratio)

        # Print analysis
        print("\nNumber of components needed for different variance thresholds:")
        print("-" * 60)
        print("Threshold    Components    Original Dimensions    Reduction (%)")
        print("-" * 60)

        # Create directory for CSV files
        os.makedirs("pca/pca_projections", exist_ok=True)

        # Process each threshold
        for threshold in thresholds:
            n_components = np.argmax(cumsum >= threshold) + 1
            reduction = (1 - n_components / len(cumsum)) * 100

            # Print current threshold analysis
            print(
                f"{threshold:^9.2f}    {n_components:^10d}    {len(cumsum):^18d}    {reduction:^13.1f}"
            )

            # Perform PCA transformation for this threshold
            transformed_data, _, _, _, _ = ourpca(X, n_components=n_components)

            # Create DataFrame with transformed data
            columns = [f"PC{i+1}" for i in range(n_components)]
            df_transformed = pd.DataFrame(transformed_data, columns=columns)

            # Save to CSV
            filename = f"pca/pca_projections/{dataset_name}_{scaling_type}_threshold_{str(threshold).replace('.','_')}.csv"
            df_transformed.to_csv(filename, index=False)
            print(f"Saved projected data to: {filename}")

    def plot_3d_scatter(
        self,
        X,
        y,
        feature_indices,
        feature_names=None,
        dataset_name=None,
        scaling_type=None,
        pca_applied=True,
    ):

        # Input validation
        if len(feature_indices) != 3:
            raise ValueError("Exactly three feature indices must be provided")

        # Extract the three features
        X_selected = X[:, feature_indices]

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"PC{i+1}" for i in feature_indices]

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot points with different colors for each class
        unique_classes = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

        for class_label, color in zip(unique_classes, colors):
            mask = y == class_label
            ax.scatter(
                X_selected[mask, 0],
                X_selected[mask, 1],
                X_selected[mask, 2],
                c=[color],
                label=f"Class {class_label}",
                alpha=0.6,
            )

        # Set labels and title
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])

        title = f"3D Scatter Plot of Selected Features\n"
        if pca_applied == True:
            if dataset_name and scaling_type:
                title += f"{dataset_name} Dataset ({scaling_type} Scaling PCA)"
        else:
            if dataset_name and scaling_type:
                title += f"{dataset_name} Dataset ({scaling_type} Scaling Without PCA)"
        ax.set_title(title)

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True)

        # Save plot
        os.makedirs("plots/3d_scatter", exist_ok=True)
        save_path = f"plots/3d_scatter/{dataset_name}_{scaling_type}_3d_scatter.jpg"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nPlot saved as: {save_path}")

        # Show plot
        plt.show()

    def plot_3d_scatter_comparison(
        self,
        X_orig,
        X_pca,
        y,
        y2,
        orig_indices,
        dataset_name=None,
        scaling_type=None,
        name=None,
        add_label=None,
    ):
        """
        Create side-by-side 3D scatter plots comparing original features with PCA components.

        Parameters:
        -----------
        X_orig : array-like
            Original data matrix
        X_pca : array-like
            PCA transformed data
        y : array-like
            Target labels for coloring
        orig_indices : list of 3 integers
            Indices of the original features to plot
        """
        if len(orig_indices) != 3:
            raise ValueError("Exactly three feature indices must be provided")

        # Extract features
        X_orig_selected = X_orig[:, orig_indices]
        X_pca_selected = X_pca[:, :3]  # Take first 3 PCs

        # Create figure with two 3D subplots
        fig = plt.figure(figsize=(20, 8))

        # Original features plot
        ax1 = fig.add_subplot(121, projection="3d")

        # PCA components plot
        ax2 = fig.add_subplot(122, projection="3d")

        # Separate noise points (-1) from cluster points
        unique_classes = np.unique(y)
        noise_mask = y == -1
        cluster_labels = unique_classes[unique_classes != -1]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cluster_labels)))

        unique_classes2 = np.unique(y2)
        noise_mask2 = y2 == -1
        cluster_labels2 = unique_classes2[unique_classes2 != -1]
        colors2 = plt.cm.rainbow(np.linspace(0, 1, len(cluster_labels2)))

        # Plot noise points first (as gray points)
        if np.any(noise_mask):
            # Original features
            ax1.scatter(
                X_orig_selected[noise_mask, 0],
                X_orig_selected[noise_mask, 1],
                X_orig_selected[noise_mask, 2],
                c="gray",
                label="Noise",
                alpha=0.3,
                marker="x",
            )
        if np.any(noise_mask2):
            # PCA components
            ax2.scatter(
                X_pca_selected[noise_mask2, 0],
                X_pca_selected[noise_mask2, 1],
                X_pca_selected[noise_mask2, 2],
                c="gray",
                label="Noise",
                alpha=0.3,
                marker="x",
            )

        # Plot clustered points
        for class_label, color in zip(cluster_labels, colors):
            mask = y == class_label

            # Original features
            ax1.scatter(
                X_orig_selected[mask, 0],
                X_orig_selected[mask, 1],
                X_orig_selected[mask, 2],
                c=[color],
                label=f"Class {class_label}",
                alpha=0.6,
            )
        for class_label, color in zip(cluster_labels2, colors2):
            mask2 = y2 == class_label
            # PCA components
            ax2.scatter(
                X_pca_selected[mask2, 0],
                X_pca_selected[mask2, 1],
                X_pca_selected[mask2, 2],
                c=[color],
                label=f"Class {class_label}",
                alpha=0.6,
            )

        # Set labels and titles
        ax1.set_xlabel(f"Feature {orig_indices[0]+1}")
        ax1.set_ylabel(f"Feature {orig_indices[1]+1}")
        ax1.set_zlabel(f"Feature {orig_indices[2]+1}")
        ax1.set_title(
            f"Original Features\n{dataset_name} Dataset ({scaling_type} Scaling {add_label})"
        )

        # Adjust the scale for original features plot
        # Get the min and max values for each dimension
        x_min, x_max = X_orig_selected[:, 0].min(), X_orig_selected[:, 0].max()
        y_min, y_max = X_orig_selected[:, 1].min(), X_orig_selected[:, 1].max()
        z_min, z_max = X_orig_selected[:, 2].min(), X_orig_selected[:, 2].max()

        # Add some padding (10%) to the ranges
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        ax1.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax1.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax1.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

        # Set equal aspect ratio for better visualization
        ax1.set_box_aspect([1, 1, 1])

        if "UMAP" in name:
            ax2.set_xlabel("UMAP1")
            ax2.set_ylabel("UMAP2")
            ax2.set_zlabel("UMAP3")
            ax2.set_title(
                f"UMAP Spaces\n{dataset_name} Dataset ({scaling_type}) Scaling {add_label}"
            )
        else:
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_zlabel("PC3")
            ax2.set_title(
                f"First 3 Principal Component \n{dataset_name} Dataset ({scaling_type}) Scaling {add_label}"
            )

        # Set equal aspect ratio for PCA plot as well
        ax2.set_box_aspect([1, 1, 1])

        # Add legends
        ax1.legend()
        ax2.legend()

        # Add grids
        ax1.grid(True)
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        os.makedirs("plots/3d_scatter", exist_ok=True)
        save_path = f"plots/3d_scatter/{dataset_name}_{scaling_type}_{name}_comparison_3d_scatter.jpg"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nPlot saved as: {save_path}")

    def align_labels(self, y_true, y_pred):
        """
        Align cluster labels to match true labels using Hungarian algorithm.

        Parameters:
        -----------
        y_true : array-like
            True class labels
        y_pred : array-like
            Predicted cluster labels

        Returns:
        --------
        array-like
            Aligned cluster labels
        """
        # Create contingency matrix
        classes = np.unique(y_true)
        clusters = np.unique(y_pred[y_pred != -1])  # Exclude noise points if any
        contingency = np.zeros((len(clusters), len(classes)))

        for i, c in enumerate(clusters):
            for j, k in enumerate(classes):
                contingency[i, j] = np.sum((y_pred == c) & (y_true == k))

        # Use Hungarian algorithm to find optimal matching
        rows, cols = linear_sum_assignment(
            -contingency
        )  # Negative for maximum matching

        # Create mapping for relabeling
        mapping = dict(zip(clusters, classes[cols]))

        # Create new labels array
        new_labels = np.copy(y_pred)
        for old_label, new_label in mapping.items():
            new_labels[y_pred == old_label] = new_label

        return new_labels

    def plot_triple_clustering_comparison(
        self, X_data, dataset_name, external_weight=0.5, y_true=None
    ):
        """
        Create three side-by-side plots comparing original PCA, K-means, and OPTICS clustering.
        """
        # Load best configurations
        if dataset_name == "cmc":
            target_k = 2
            thresholdk = 0.8
            thresholdo = 0.9
            target_ko = 3
        elif dataset_name == "pen-based":
            target_k = 9
            thresholdk = 0.8
            thresholdo = 0.8
            target_ko = 11

        # Apply PCA with this threshold
        transformed_pca, _, _, _, _ = ourpca(X_data, threshold=thresholdk)
        X_pcak = transformed_pca

        transformed_pcao, _, _, _, _ = ourpca(X_data, threshold=thresholdo)
        X_pcao = transformed_pcao
        # Apply PCA with the threshold from configurations
        transformed_pca, eigenvalues_our, eigenvectors_our, cov_matrix, _ = ourpca(
            X_data, n_components=3
        )
        X_pca = transformed_pca

        # Apply custom K-means with best configuration
        kmeans = KMeans()
        centroids, kmeans_labels, _ = kmeans.k_means(
            X=X_pcak,
            K=target_k,
            distance_metric="cosine",
            max_iters=100,
            plus_plus=True,
            random_state=42,
        )

        # Apply OPTICS with best configuration
        if dataset_name == "pen-based":
            optics = OPTICS(
                min_samples=11,
                max_eps=2.50,
                metric="euclidean",
                algorithm="brute",
                cluster_method="xi",
                xi=0.0575,
                n_jobs=-1,
            )

        else:

            optics = OPTICS(
                min_samples=35,
                max_eps=1.5,
                metric="euclidean",
                algorithm="brute",
                n_jobs=-1,
            )

        optics_labels = optics.fit_predict(X_pcao)

        if dataset_name == "pen-based":
            max_classes = max(len(np.unique(y_true)), target_k, 11)
        elif dataset_name == "cmc":
            max_classes = max(len(np.unique(y_true)), target_k, 3)
        colors = plt.cm.rainbow(np.linspace(0, 1, max_classes))

        fig = plt.figure(figsize=(24, 8))

        # Modified original PCA plot to show true categories
        ax1 = fig.add_subplot(131, projection="3d")
        unique_true = np.unique(y_true)
        for i, label in enumerate(unique_true):
            mask = y_true == label
            ax1.scatter(
                X_pcak[mask, 0],
                X_pcak[mask, 1],
                X_pcak[mask, 2],
                c=[colors[i]],
                alpha=0.6,
                label=f"Class {label}",
            )
        ax1.set_title(f"Original PCA with True Labels\n{dataset_name} Dataset")

        # K-means clustering plot
        ax2 = fig.add_subplot(132, projection="3d")
        unique_kmeans = np.unique(kmeans_labels)
        for i, label in enumerate(unique_kmeans):
            mask = kmeans_labels == label
            ax2.scatter(
                X_pcak[mask, 0],
                X_pcak[mask, 1],
                X_pcak[mask, 2],
                c=[colors[i]],
                alpha=0.6,
                label=f"Cluster {label}",
            )
        ax2.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="black",
            marker="x",
            s=200,
            linewidth=3,
            label="Centroids",
        )
        ax2.set_title(f"PCA with K-means++\n(k={target_k})")

        # OPTICS clustering plot
        ax3 = fig.add_subplot(133, projection="3d")
        non_noise_mask = optics_labels != -1
        unique_optics = np.unique(optics_labels[non_noise_mask])
        for i, label in enumerate(unique_optics):
            mask = optics_labels == label
            color_idx = i % len(colors)
            ax3.scatter(
                X_pcao[mask, 0],
                X_pcao[mask, 1],
                X_pcao[mask, 2],
                c=[colors[color_idx]],
                alpha=0.6,
                label=f"Cluster {label}",
            )
        if not all(non_noise_mask):
            ax3.scatter(
                X_pcao[~non_noise_mask, 0],
                X_pcao[~non_noise_mask, 1],
                X_pcao[~non_noise_mask, 2],
                c="gray",
                alpha=0.3,
                label="Noise",
            )
        ax3.set_title(f"PCA with OPTICS \n(k={target_ko})")

        # Set labels for all plots
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.legend()

        plt.tight_layout()

        # Save plot
        os.makedirs("plots/clustering_comparison", exist_ok=True)
        save_path = f"plots/clustering_comparison/{dataset_name}_triple_comparison.jpg"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\nPlot saved as: {save_path}")

    def plot_triple_clustering_comparison_umap(
        self, X_data, dataset_name, external_weight=0.5, y_true=None
    ):
        """
        Create three side-by-side plots comparing original UMAP, K-means, and OPTICS clustering.
        """
        # Load best configurations
        if dataset_name == "cmc":
            target_k = 2
            target_ko = 3
            max_classes = max(len(np.unique(y_true)), target_k, 11)
        elif dataset_name == "pen-based":
            target_k = 9
            target_ko = 11
            max_classes = max(len(np.unique(y_true)), target_k, 5)

        # UMAP transformation
        import umap

        umap_reducer = umap.UMAP(n_components=3, random_state=42)
        umap_embedded = umap_reducer.fit_transform(X_data)

        colors = plt.cm.rainbow(np.linspace(0, 1, max_classes))

        # Apply custom K-means with best configuration
        kmeans = KMeans()
        centroids, kmeans_labels, _ = kmeans.k_means(
            X=umap_embedded,
            K=target_k,
            distance_metric="cosine",
            max_iters=100,
            plus_plus=True,
            random_state=42,
        )

        if dataset_name == "pen-based":
            optics_umap = OPTICS(
                min_samples=150,
                max_eps=0.5,
                metric="cosine",
                algorithm="auto",
                xi=0.15,
                n_jobs=-1,
            )

        else:
            optics_umap = OPTICS(
                min_samples=51, max_eps=1.2, metric="manhattan", algorithm="auto"
            )

        optics_labels = optics_umap.fit_predict(umap_embedded)
        # Align cluster labels with true labels
        unique_true = np.unique(y_true)
        unique_kmeans = np.unique(kmeans_labels)
        unique_optics = np.unique(optics_labels[optics_labels != -1])

        # Get the maximum number of clusters across all methods

        colors = plt.cm.rainbow(np.linspace(0, 1, max_classes))
        # Color mapping setup
        max_classes = max(len(np.unique(y_true)), target_k)
        colors = plt.cm.rainbow(np.linspace(0, 1, max_classes))

        fig = plt.figure(figsize=(24, 8))

        # Original UMAP plot with true categories
        ax1 = fig.add_subplot(131, projection="3d")
        unique_true = np.unique(y_true)
        for i, label in enumerate(unique_true):
            mask = y_true == label
            ax1.scatter(
                umap_embedded[mask, 0],
                umap_embedded[mask, 1],
                umap_embedded[mask, 2],
                c=[colors[i]],
                alpha=0.6,
                label=f"Class {label}",
            )
        ax1.set_title(f"Original UMAP with True Labels\n{dataset_name} Dataset")

        # K-means clustering plot
        ax2 = fig.add_subplot(132, projection="3d")
        unique_kmeans = np.unique(kmeans_labels)
        for i, label in enumerate(unique_kmeans):
            mask = kmeans_labels == label
            ax2.scatter(
                umap_embedded[mask, 0],
                umap_embedded[mask, 1],
                umap_embedded[mask, 2],
                c=[colors[i]],
                alpha=0.6,
                label=f"Cluster {label}",
            )
        ax2.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="black",
            marker="x",
            s=200,
            linewidth=3,
            label="Centroids",
        )
        ax2.set_title(f"UMAP with K-means++\n(k={target_k})")

        # OPTICS clustering plot
        ax3 = fig.add_subplot(133, projection="3d")
        non_noise_mask = optics_labels != -1
        unique_optics = np.unique(optics_labels[non_noise_mask])
        for i, label in enumerate(unique_optics):
            mask = optics_labels == label
            color_idx = (
                int(label) if int(label) < len(colors) else 0
            )  # Ensure color index is in bounds
            ax3.scatter(
                umap_embedded[mask, 0],
                umap_embedded[mask, 1],
                umap_embedded[mask, 2],
                c=[colors[color_idx]],
                alpha=0.6,
                label=f"Cluster {label}",
            )
        if not all(non_noise_mask):
            ax3.scatter(
                umap_embedded[~non_noise_mask, 0],
                umap_embedded[~non_noise_mask, 1],
                umap_embedded[~non_noise_mask, 2],
                c="gray",
                alpha=0.3,
                label="Noise",
            )
        ax3.set_title(f"UMAP with OPTICS \n (k={target_ko})")

        # Set labels for all plots
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_zlabel("UMAP3")
            ax.legend()

        plt.tight_layout()

        # Save plot
        os.makedirs("plots/clustering_comparison", exist_ok=True)
        save_path = (
            f"plots/clustering_comparison/{dataset_name}_triple_comparison_umap.jpg"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\nPlot saved as: {save_path}")

    def plot_kmeans_optics_comparison(
        self, X_data, kmeans_labels, optics_labels, feature_indices, 
        dataset_name=None, scaling_type=None
    ):
        """
        Create side-by-side 3D scatter plots comparing K-means++ and OPTICS clustering
        on the same original features.

        Parameters:
        -----------
        X_data : array-like
            Original data matrix
        kmeans_labels : array-like
            Cluster labels from K-means++
        optics_labels : array-like
            Cluster labels from OPTICS
        feature_indices : list of 3 integers
            Indices of the features to plot
        dataset_name : str
            Name of the dataset for plot titles
        scaling_type : str
            Type of scaling applied to the data
        """
        if len(feature_indices) != 3:
            raise ValueError("Exactly three feature indices must be provided")

        # Extract selected features
        X_selected = X_data[:, feature_indices]

        # Create figure with two 3D subplots
        fig = plt.figure(figsize=(20, 8))

        # K-means plot
        ax1 = fig.add_subplot(121, projection="3d")
        # OPTICS plot
        ax2 = fig.add_subplot(122, projection="3d")

        # Setup for K-means plot
        unique_kmeans = np.unique(kmeans_labels)
        colors_kmeans = plt.cm.rainbow(np.linspace(0, 1, len(unique_kmeans)))

        # Setup for OPTICS plot
        noise_mask = optics_labels == -1
        unique_optics = np.unique(optics_labels[optics_labels != -1])
        colors_optics = plt.cm.rainbow(np.linspace(0, 1, len(unique_optics)))

        # Plot K-means clusters
        for i, label in enumerate(unique_kmeans):
            mask = kmeans_labels == label
            ax1.scatter(
                X_selected[mask, 0],
                X_selected[mask, 1],
                X_selected[mask, 2],
                c=[colors_kmeans[i]],
                label=f"Cluster {label}",
                alpha=0.6
            )

        # Plot OPTICS clusters and noise
        if np.any(noise_mask):
            ax2.scatter(
                X_selected[noise_mask, 0],
                X_selected[noise_mask, 1],
                X_selected[noise_mask, 2],
                c='gray',
                label='Noise',
                alpha=0.3,
                marker='x'
            )
        
        for i, label in enumerate(unique_optics):
            mask = optics_labels == label
            ax2.scatter(
                X_selected[mask, 0],
                X_selected[mask, 1],
                X_selected[mask, 2],
                c=[colors_optics[i]],
                label=f"Cluster {label}",
                alpha=0.6
            )

        # Get plot limits
        x_min, x_max = X_selected[:, 0].min(), X_selected[:, 0].max()
        y_min, y_max = X_selected[:, 1].min(), X_selected[:, 1].max()
        z_min, z_max = X_selected[:, 2].min(), X_selected[:, 2].max()

        # Add padding
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Apply same limits to both plots
        for ax in [ax1, ax2]:
            ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
            ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
            ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
            
            # Set labels
            ax.set_xlabel(f"Feature {feature_indices[0]+1}")
            ax.set_ylabel(f"Feature {feature_indices[1]+1}")
            ax.set_zlabel(f"Feature {feature_indices[2]+1}")
            
            # Add grid and set aspect ratio
            ax.grid(True)
            ax.set_box_aspect([1, 1, 1])

        # Set titles
        ax1.set_title(f"K-means++ Clustering\n{dataset_name} Dataset ({scaling_type} Scaling)")
        ax2.set_title(f"OPTICS Clustering\n{dataset_name} Dataset ({scaling_type} Scaling)")

        # Add legends
        ax1.legend(title=f"K-means++ (k={len(unique_kmeans)})")
        ax2.legend(title=f"OPTICS (k={len(unique_optics)})")

        plt.tight_layout()

        # Save plot
        os.makedirs("plots/clustering_comparison", exist_ok=True)
        save_path = f"plots/clustering_comparison/{dataset_name}_{scaling_type}_kmeans_optics_comparison.jpg"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nPlot saved as: {save_path}")

    def save_principal_components_composition(
        self, eigenvectors, dataset_name, scaling_type, n_components=3
    ):
        """
        Save the top contributing features to PCA components.
        """
        # Calculate total contribution of each feature
        total_contributions = np.sum(np.abs(eigenvectors[:, :n_components]), axis=1)

        # Get indices of top contributing features
        top_features_idx = np.argsort(total_contributions)[::-1][:n_components]

        # Create simple DataFrame with results
        df = pd.DataFrame(
            {
                "Feature": [f"Feature_{idx+1}" for idx in top_features_idx],
                "Total_Contribution": total_contributions[top_features_idx],
            }
        )

        # Save to CSV
        os.makedirs("pca/pca_projections", exist_ok=True)
        filename = f"pca/pca_projections/{dataset_name}_{scaling_type}_top_features.csv"
        df.to_csv(filename, index=False)

        # Print results
        print(f"\nTop {n_components} contributing features:")
        print(df.to_string(index=False))
