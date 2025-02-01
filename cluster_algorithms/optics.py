import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.calibration import LabelEncoder
from sklearn.cluster import OPTICS
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    fowlkes_mallows_score,
    calinski_harabasz_score,
    silhouette_score,
    normalized_mutual_info_score,
)


class Optics:

    def __init__(
        self,
        min_samples=5,
        max_eps=np.inf,
        metric="euclidean",
        algorithm="auto",
        pca_threshold=None,
    ):
        """
        Initialize the OPTICS clustering model with default parameters.
        """
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.algorithm = algorithm
        self.model = None
        self.labels_ = None
        self.pca_threshold = pca_threshold

    def fit(self, X):
        """
        Fit the OPTICS model to the data.
        """
        self.model = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=-1,
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def evaluate(self, X, y_true, scaling_method):
        """
        Evaluate the clustering performance and compute various metrics.
        """
        # Exclude noise points
        mask = self.labels_ != -1
        y_pred_filtered = self.labels_[mask]
        y_true_filtered = y_true[mask]
        X_filtered = X[mask]

        n_clusters = len(set(y_pred_filtered))
        n_noise = np.sum(self.labels_ == -1)

        # Compute metrics
        metrics = {}
        metrics["Dataset scaling"] = scaling_method
        if n_clusters > 1 and n_clusters < len(y_pred_filtered):
            metrics["Silhouette Score"] = silhouette_score(X_filtered, y_pred_filtered)
            metrics["Davies-Bouldin Index"] = davies_bouldin_score(
                X_filtered, y_pred_filtered
            )
            metrics["Calinski-Harabasz Score"] = calinski_harabasz_score(
                X_filtered, y_pred_filtered
            )
            metrics["Adjusted Rand Index"] = adjusted_rand_score(
                y_true_filtered, y_pred_filtered
            )
            metrics["Purity"] = self.purity_score(y_true_filtered, y_pred_filtered)
            metrics["Fowlkes-Mallows Index"] = fowlkes_mallows_score(
                y_true_filtered, y_pred_filtered
            )
            metrics["Normalized Mutual Info"] = normalized_mutual_info_score(
                y_true_filtered, y_pred_filtered
            )

        else:
            metrics["Silhouette Score"] = np.nan
            metrics["Davies-Bouldin Index"] = np.nan
            metrics["Calinski-Harabasz Score"] = np.nan
            metrics["Adjusted Rand Index"] = np.nan
            metrics["Purity"] = np.nan
            metrics["Fowlkes-Mallows Index"] = np.nan
            metrics["Normalized Mutual Info"] = np.nan

        metrics["Number of Clusters"] = n_clusters

        return metrics, n_clusters

    @staticmethod
    def purity_score(y_true, y_pred):
        """
        Calculate the purity score for the clustering results.
        """
        contingency_matrix = confusion_matrix(y_true, y_pred)
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
            contingency_matrix
        )
        return purity

    def perform_optics(self, dataset_name, dataset_choice, X, y_true):

        if dataset_choice == 1:
            scaling_method = "MinMax"
        elif dataset_choice == 2:
            scaling_method = "Robust"
        elif dataset_choice == 3:
            scaling_method = "Standard"
        if dataset_name == "cmc":
            min_samples_list = [
                25,
                35,
                45,
                52,
                55,
                57,
            ]
            max_eps_list = [1.5, 3.0, 5.0, 7.0, 9.0]
            gt_n_classes = 3
        elif dataset_name == "hepatitis":
            min_samples_list = [
                3,
                4,
                5,
                6,
                7,
            ]
            max_eps_list = [0.5, 0.7, 1.0, 1.5, 2.0]
            gt_n_classes = 2
        elif dataset_name == "pen-based":
            gt_n_classes = 10
            min_samples_list = [
                15,
                20,
                23,
                30,
                32,
                40,
            ]
            max_eps_list = [1.0, 3.0, 5.0, 7.0, 9.0]
        elif dataset_name == None:
            print("Please provide a valid dataset name")
            return "Please provide a valid dataset name"

        metric_list = [
            "euclidean",
            "manhattan",
            "cosine",
        ]
        algorithm_list = ["auto", "brute"]
        results = []
        start_time = time.time()
        # Encode labels if they are not numeric
        if y_true.dtype == object or y_true.dtype == "str":
            le = LabelEncoder()
            y_true_encoded = le.fit_transform(y_true)
        else:
            y_true_encoded = y_true.astype(int)

        X_reduced = X

        for min_samples in min_samples_list:
            for max_eps in max_eps_list:
                for metric in metric_list:
                    for algorithm in algorithm_list:
                        print(
                            f"\nRunning OPTICS with min_samples={min_samples}, metric='{metric}', algorithm='{algorithm}'"
                        )

                        self.model = OPTICS(
                            min_samples=min_samples,
                            max_eps=max_eps,
                            metric=metric,
                            xi=0.05,
                            algorithm=algorithm,
                            n_jobs=-10,
                        )
                        print(self.model)
                        print(
                            f"Model parameters: min_samples={self.model.min_samples}, max_eps={self.model.max_eps}, metric={self.model.metric}, algorithm={self.model.algorithm}"
                        )
                        # Fit the model
                        self.labels_ = self.model.fit_predict(X_reduced)
                        unique_labels = np.unique(self.labels_)
                        print(unique_labels)
                        # Evaluate the clustering
                        metrics, n_clusters = self.evaluate(
                            X_reduced, y_true_encoded, scaling_method
                        )

                        # Print metrics
                        for metric_name, metric_value in metrics.items():
                            print(f"{metric_name}: {metric_value}")
                        if n_clusters <= 1:
                            pass
                        else:
                            # Collect results
                            result = {
                                "Dataset": dataset_name,
                                "min_samples": min_samples,
                                "metric": metric,
                                "algorithm": algorithm,
                                "max_epsilon": max_eps,
                                **metrics,
                            }
                            results.append(result)

        # Track time
        end_time = time.time()
        print(
            f"\nTotal computation time for dataset {dataset_name}: {end_time - start_time:.2f} seconds"
        )

        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        relative_dir = Path("results/optic_results")
        relative_dir.mkdir(parents=True, exist_ok=True)

        if self.pca_threshold:
            output_file_path = (
                relative_dir
                / f"optics_results_{dataset_name}_{scaling_method}_threshold_{str(self.pca_threshold).replace('.','_')}.csv"
            )
        else:
            output_file_path = (
                relative_dir / f"optics_results_{dataset_name}_{scaling_method}.csv"
            )

        results_df.to_csv(output_file_path, index=False)

        print("\nAll Results:")
        print(results_df)
