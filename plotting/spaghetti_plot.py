import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class CreatesSpaghettiPlot:
    def __init__(self, base_path="results", algorithm="K-Means"):
        self.base_path = base_path
        self.algorithm = algorithm

    def read_dataset(self, dataset_name):
        """
        Read and combine CSV files for a given dataset
        """
        if self.algorithm in ["K-Means", "K-Means++"]:
            files = [
                f"{self.base_path}/k_means_results/{dataset_name}_MinMax_{self.algorithm}_metrics.csv",
                f"{self.base_path}/k_means_results/{dataset_name}_Robust_{self.algorithm}_metrics.csv",
                f"{self.base_path}/k_means_results/{dataset_name}_Standard_{self.algorithm}_metrics.csv",
            ]
        elif self.algorithm == "G-Means":
            files = [
                f"{self.base_path}/g_means_results/{dataset_name}_MinMax_{self.algorithm}_metrics.csv",
                f"{self.base_path}/g_means_results/{dataset_name}_Robust_{self.algorithm}_metrics.csv",
                f"{self.base_path}/g_means_results/{dataset_name}_Standard_{self.algorithm}_metrics.csv",
            ]

        elif self.algorithm == "C-Means":
            files = [
                f"{self.base_path}/c_means_results/{dataset_name}_0.5.csv",
            ]

        dataframes = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                if self.algorithm == "C-Means":
                    # Extract weight from filename
                    weight = file_path.split("_")[-1].replace(".csv", "")
                    df["Dataset scaling"] = f"Weight_{weight}"  # e.g., "Weight_0.25"
                else:
                    # Original scaling type extraction
                    scaling = file_path.split("_")[-3]
                    df["Dataset scaling"] = scaling
                dataframes.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")

        if not dataframes:
            raise ValueError("No data files were successfully loaded")

        return pd.concat(dataframes, ignore_index=True)

    def get_k_range(self, dataset_name):
        if dataset_name == "cmc":
            return range(2, 9)  # [2, 3, 4, 5, 6, 7, 8]
        elif dataset_name == "hepatitis":
            return range(2, 9)  # [2, 3, 4, 5, 6, 7, 8]
        elif dataset_name == "pen-based":
            return range(8, 13)  # [8, 9, 10, 11, 12]
        return None

    def create_visualization(self, dataset_name, output_name=None):
        """
        Create visualization adapted for all algorithms (K-means, G-means, and C-means)
        """
        combined_data = self.read_dataset(dataset_name)

        if output_name is None:
            output_name = f"Spagetti_plots_of_metrics_{dataset_name}_{self.algorithm}"

        # Handle different algorithms' data structures
        if "Optimal k" in combined_data.columns:
            combined_data["K"] = combined_data["Optimal k"]
        elif self.algorithm == "C-Means":
            combined_data["K"] = combined_data["num_clusters"]

        # Define metrics based on algorithm
        if self.algorithm == "C-Means":
            metrics_info = {
                "silhouette_score": ("#0d47a1", "native"),
                "davies_bouldin": ("#1976d2", "invert"),
                "calinski_harabasz": ("#64b5f6", "zero_one"),
                "partition_coefficient": ("#e65100", "zero_one"),
                "partition_entropy": ("#fb8c00", "invert"),
                "xie_beni_index": ("#ffb74d", "invert"),
            }
        else:
            metrics_info = {
                "silhouette_score": ("#0d47a1", "native"),
                "davies_bouldin": ("#1976d2", "invert"),
                "calinski_harabasz": ("#64b5f6", "zero_one"),
                "adjusted_rand_index": ("#e65100", "native"),
                "fowlkes_mallows_index": ("#fb8c00", "zero_one"),
                "purity_score": ("#ffb74d", "zero_one"),
            }

        # Scale the data
        scaled_data = combined_data.copy()
        for metric, (_, scale_type) in metrics_info.items():
            if scale_type == "native":
                continue
            else:
                min_val = combined_data[metric].min()
                max_val = combined_data[metric].max()
                scaled_data[metric] = (combined_data[metric] - min_val) / (
                    max_val - min_val
                )
                if scale_type == "invert":
                    scaled_data[metric] = 1 - scaled_data[metric]

        # For C-means, create single plot
        if self.algorithm == "C-Means":
            plt.figure(figsize=(12, 8))

            for metric, (color, _) in metrics_info.items():
                plt.plot(
                    scaled_data["K"],
                    scaled_data[metric],
                    marker="o",
                    label=metric,
                    color=color,
                )

            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Normalized Score")
            plt.title(
                f"Best C-Means Results for {dataset_name}\n(Inverted scores where appropriate)"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            k_values = list(self.get_k_range(dataset_name))
            plt.xticks(k_values)
            plt.ylim(0.0, 1.1)

        else:
            # Original 3x3 grid for K-means and G-means
            fig, axes = plt.subplots(3, 3, figsize=(15, 17))
            fig.suptitle(
                f"Clustering Results for {dataset_name} using {self.algorithm} (Davies Bouldin Score Inverted)",
                fontsize=16,
                y=1.02,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            standardizations = sorted(scaled_data["Dataset scaling"].unique())
            distance_metrics = sorted(scaled_data["distance_metric"].unique())
            k_values = list(
                self.get_k_range(dataset_name)
            )  # Get K values for this dataset

            for i, standard in enumerate(standardizations):
                for j, distance in enumerate(distance_metrics):
                    ax = axes[i, j]
                    subset = scaled_data[
                        (scaled_data["distance_metric"] == distance)
                        & (scaled_data["Dataset scaling"] == standard)
                    ]

                    for metric, (color, _) in metrics_info.items():
                        if len(subset) > 0:
                            ax.plot(
                                subset["K"],
                                subset[metric],
                                marker="o",
                                label=metric,
                                color=color,
                            )

                    ax.set_xlabel("Number of Clusters (K)")
                    ax.set_ylabel("Normalized Score")
                    ax.set_xticks(k_values)
                    ax.set_xticklabels(k_values)
                    ax.set_title(f"{standard} - {distance}")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.1)

                    if i == 0 and j == 0:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()

        # Save the plot
        output_dir = Path("plots/spagetti")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{output_name}.png"

        save_path = output_dir / output_filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to: {save_path.absolute()}")

        return None
