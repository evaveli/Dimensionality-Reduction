import numpy as np
from scipy.optimize import linear_sum_assignment
from plotting.plot_confusion import plot_confusion
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from .distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)
from utils import print_divider


def compute_sse(X, centroids, labels, distance_metric):
    """
    Compute the Sum of Squared Errors (SSE) for clustering.
    """
    sse = 0.0
    for k in range(len(centroids)):
        cluster_data = X[labels == k]
        if cluster_data.shape[0] > 0:
            if distance_metric == "euclidean":
                distances = euclidean_distance(cluster_data, centroids[k])
            elif distance_metric == "manhattan":
                distances = manhattan_distance(cluster_data, centroids[k])
            elif distance_metric == "cosine":
                distances = cosine_distance(cluster_data, centroids[k])
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")
            sse += np.sum(distances**2)
    return sse


class MetricCollection:
    """
    A class to evaluate clustering performance using internal and external indexes.
    """

    def __init__(self, X, best_labels, metric, y_true_filtered, y_pred_filtered):
        """
        Initialize the MetricCollection with clustering results and metadata.
        """
        self.X = X
        self.best_labels = best_labels
        self.metric = metric
        self.y_true_filtered = y_true_filtered
        self.y_pred_filtered = y_pred_filtered
        self.y_pred_mapped = None
        self.silhouette = None
        self.davies_bouldin = None
        self.calinski_harabasz = None
        self.ari = None
        self.fmi = None
        self.nmi = None
        self.purity = None

    def map_labels(self):
        """
        Map predicted labels to true labels using the Hungarian algorithm for optimal label assignment.
        """
        labels_true = np.unique(self.y_true_filtered)
        labels_pred = np.unique(self.y_pred_filtered)
        labels = np.union1d(labels_true, labels_pred)
        cm = confusion_matrix(self.y_true_filtered, self.y_pred_filtered, labels=labels)

        # Find the best label mapping
        row_idx, col_idx = linear_sum_assignment(-cm)
        label_mapping = {labels[col]: labels[row] for row, col in zip(row_idx, col_idx)}

        # Map predicted labels to true labels
        self.y_pred_mapped = np.vectorize(lambda x: label_mapping.get(x, x))(
            self.y_pred_filtered
        )

    def compute_confusion(
        self,
        K,
        metric,
        dataset_name,
        scaling_method,
        algo,
        plus_plus=False,
    ):
        """
        Compute and output a confusion matrix for the clustering results.
        """

        # Output confusion matrix to a file
        cm_mapped = confusion_matrix(self.y_true_filtered, self.y_pred_mapped)
        plot_confusion(cm_mapped, K, metric, dataset_name, scaling_method, algo)

    def compute_purity_score(self):
        """
        Calculate the purity score for the clustering results.
        """
        contingency_matrix = confusion_matrix(self.y_true_filtered, self.y_pred_mapped)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def calculate_internal_indexes(self):
        """
        Calculate internal clustering metrics: Silhouette, Davies-Bouldin, and Calinski-Harabasz.
        """
        if len(np.unique(self.best_labels)) > 1:  # Ensure at least two clusters exist
            self.silhouette = silhouette_score(
                self.X, self.best_labels, metric=self.metric
            )
            self.davies_bouldin = davies_bouldin_score(self.X, self.best_labels)
            self.calinski_harabasz = calinski_harabasz_score(self.X, self.best_labels)

    def calculate_external_indexes(self):
        """
        Calculate external clustering metrics: Adjusted Rand Index, Fowlkes-Mallows Index,
        Normalized Mutual Information, and Purity score.
        """
        if len(np.unique(self.y_pred_filtered)) > 1 and len(self.y_pred_filtered) > 1:
            self.ari = adjusted_rand_score(self.y_true_filtered, self.y_pred_filtered)
            self.fmi = fowlkes_mallows_score(self.y_true_filtered, self.y_pred_filtered)
            self.nmi = normalized_mutual_info_score(
                self.y_true_filtered, self.y_pred_filtered
            )
            self.purity = self.compute_purity_score()

    def print_evaluation(self, K=None, *additional_lines):
        """
        Print clustering evaluation results.
        """
        for line in additional_lines:
            print(line)

        print_divider(before="\n")

        if not K == None:
            print(
                f"Evaluation results for K = {K} and distance metric: '{self.metric}':"
            )

        print(
            f"Silhouette Score: \t\t{self.silhouette}",
            f"Davies Bouldin Score: \t\t{self.davies_bouldin}",
            f"Calinski Harabasz Score: \t{self.calinski_harabasz}",
            f"Adjusted Rand Index: \t\t{self.ari}",
            f"Fowlkes-Mallows Index: \t\t{self.fmi}",
            f"Normalized Mutual Information: \t{self.nmi}",
            f"Purity Score: \t\t\t{self.purity}",
            sep="\n",
        )

        print_divider(after="\n")
