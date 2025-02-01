import os
import numpy as np
from matplotlib import pyplot as plt
from .k_means_plus_plus import KMeansPlusPlus
from metrics.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)
from metrics.evaluation_metrics import compute_sse, MetricCollection
from plotting.plot_elbow import plot_elbow
from utils import save_metrics_to_csv, print_heading2


class KMeans:

    def plot_clusters(self, X, labels, centroids, title):
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100)
        plt.title(title)
        plt.show()

    def create_results_dict(
        self,
        results,
        metric,
        K,
        best_sse,
        initial_centroid_idxs,
        metric_collection,
        scaling_method,
        clustering_algo,
        threshold,
    ):
        results[metric][K] = {
            "Dataset scaling": f"{scaling_method}_PCA_{threshold}",
            "Method name": clustering_algo,
            "sse": float(best_sse),
            "centroids": (
                initial_centroid_idxs.tolist()
                if isinstance(initial_centroid_idxs, np.ndarray)
                else initial_centroid_idxs
            ),
            "silhouette_score": float(metric_collection.silhouette),
            "davies_bouldin": float(metric_collection.davies_bouldin),
            "calinski_harabasz": float(metric_collection.calinski_harabasz),
            "adjusted_rand_index": float(metric_collection.ari),
            "fowlkes_mallows_index": float(metric_collection.fmi),
            "normalized_mutual_info": float(metric_collection.nmi),
            "purity_score": float(metric_collection.purity),
        }
        return results

    def k_means(
        self,
        X,
        K,
        distance_metric="cosine",
        max_iters=100,
        tol=1e-6,
        plus_plus=False,
        random_state=None,
    ):
        """
        K-Means clustering algorithm.
        """
        # If random_state is provided, create a separate RandomState object
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.RandomState()

        n_samples, n_features = X.shape

        if plus_plus:
            kmeans_plus_plus = KMeansPlusPlus(n_clusters=K, d_metric=distance_metric)
            centroids = kmeans_plus_plus.get_controids(X, n_samples, random_state)
            initial_centroid_indices = None  # indicate that we used k-means++
        else:
            # Initialize centroids by selecting K random samples from X
            initial_centroid_indices = rng.choice(n_samples, K, replace=False)
            centroids = X[initial_centroid_indices]

        for iteration in range(max_iters):
            # Assignment Step
            # Compute distances between each point and each centroid
            if distance_metric == "euclidean":
                distances = np.array(
                    [euclidean_distance(X, centroid) for centroid in centroids]
                ).T
            elif distance_metric == "manhattan":
                distances = np.array(
                    [manhattan_distance(X, centroid) for centroid in centroids]
                ).T
            elif distance_metric == "cosine":
                distances = np.array(
                    [cosine_distance(X, centroid) for centroid in centroids]
                ).T
            else:
                raise ValueError("Unsupported distance metric.")

            # Assign labels based on closest centroid
            labels = np.argmin(distances, axis=1)

            # Store the old centroids for convergence check
            centroids_old = centroids.copy()

            # Update Step
            # Recalculate centroids as the mean of assigned points
            for k in range(K):
                # Check if any points are assigned to centroid k
                if np.any(labels == k):
                    centroids[k] = X[labels == k].mean(axis=0)
                else:
                    # Handle empty clusters by reinitializing centroid
                    centroids[k] = X[np.random.choice(n_samples)]

            # Convergence Check
            centroid_shifts = np.linalg.norm(centroids - centroids_old, axis=1)
            if np.all(centroid_shifts <= tol):
                print(f"Converged in {iteration+1} iterations.")
                break
        return centroids, labels, initial_centroid_indices

    def run_k_means_experiments(
        self,
        X,
        dataset_choice,
        clustering_algo,
        y_true,
        plus_plus=False,
        dataset_name=None,
        random_state=None,
        threshold=None,
    ):

        scaling_method = "Robust" if dataset_name == "cmc" else "MinMax"
        if dataset_name == "cmc":
            K_values = [2, 3, 4, 5, 6, 7, 8]
            max_iters = 100
            gt_n_classes = 3
            n_initialization = 15
        elif dataset_name == "hepatitis":
            K_values = [2, 3, 4, 5, 6, 7, 8]
            max_iters = 100
            gt_n_classes = 2
            n_initialization = 25
        elif dataset_name == "pen-based":
            K_values = [8, 9, 10, 11, 12]
            gt_n_classes = 10
            max_iters = 200
            n_initialization = 10
        elif dataset_name == None:
            print("Please provide a valid dataset name")
            return "Please provide a valid dataset name"
        # Adjust K values as needed
        distance_metrics = ["euclidean", "manhattan", "cosine"]
        results = {}

        for metric in distance_metrics:
            results[metric] = {}
            for K in K_values:
                print_heading2(f"Running K-Means Analysis for K: {K}, metric: {metric}")
                best_sse = float("inf")
                best_labels = None
                best_centroids = None

                # Run K-Means n times to eliminate bad seeding risk
                base_seed = 42
                for i in range(n_initialization):
                    print(f"\nIteration number: {i+1}")
                    iteration_seed = (
                        None if random_state is None else (base_seed + i * 1000)
                    )

                    centroids, labels, initial_centroid_indices = self.k_means(
                        X,
                        K=K,
                        distance_metric=metric,
                        plus_plus=plus_plus,
                        max_iters=max_iters,
                        random_state=iteration_seed,
                    )

                    print(f"Initial centroid indices: {initial_centroid_indices}")
                    sse = compute_sse(X, centroids, labels, metric)
                    print(f"SSE:{sse}")

                    # Save best run
                    if sse < best_sse:
                        best_sse = sse
                        best_labels = labels
                        best_centroids = centroids

                y_true_filtered = np.array(y_true, dtype=int)
                y_pred_filtered = np.array(best_labels, dtype=int)

                # Create a new MetricCollection object
                metric_collection = MetricCollection(
                    X=X,
                    best_labels=best_labels,
                    metric=metric,
                    y_true_filtered=y_true_filtered,
                    y_pred_filtered=y_pred_filtered,
                )

                metric_collection.map_labels()  # Map labels for confusion matrix and purity

                # Compute confusion matrix only when number of clusters corresponds to the true value
                if K == gt_n_classes:
                    metric_collection.compute_confusion(
                        K,
                        metric,
                        dataset_name,
                        scaling_method,
                        clustering_algo,
                        plus_plus,
                    )

                metric_collection.calculate_internal_indexes()
                metric_collection.calculate_external_indexes()

                additional_lines = [
                    f"Best SSE for K={K}, metric='{metric}': {best_sse}"
                ]
                metric_collection.print_evaluation(K, *additional_lines)

                results = self.create_results_dict(
                    results,
                    metric,
                    K,
                    best_sse,
                    initial_centroid_indices,
                    metric_collection,
                    scaling_method,
                    clustering_algo,
                    threshold,
                )

        plot_elbow(
            results,
            dataset_name=dataset_name,
            scaling_method=scaling_method,
            clustering_algo=clustering_algo,
        )

        best_results = {}
        for metric in distance_metrics:
            best_results[metric] = {}
            for K in K_values:
                best_k_result = results[metric][K]
                best_results[metric][K] = best_k_result
