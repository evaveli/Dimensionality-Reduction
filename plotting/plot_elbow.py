import os
import matplotlib.pyplot as plt
from utils import create_directory


def plot_elbow(
    results,
    dataset_name=None,
    scaling_method=None,
    clustering_algo=None,
):
    metrics = results.keys()

    for metric in metrics:
        ks = list(results[metric].keys())
        sses = [results[metric][k]["sse"] for k in ks]

        plt.figure(figsize=(8, 6))
        plt.plot(ks, sses, marker="o", label=f"{metric.capitalize()}")
        plt.title(f"Elbow Method for {metric.capitalize()} Metric")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.grid(True)
        plt.legend()
        plt.xticks(ks)

        if dataset_name and scaling_method and clustering_algo:
            # Create dir if not existing
            plot_dir = os.path.join("plots", "elbows")
            create_directory(plot_dir)

            file_path = os.path.join(
                plot_dir,
                f"elbow_{dataset_name}_{scaling_method}_{clustering_algo}_{metric}",
            )

            plt.savefig(file_path)
            print(f"Plot saved as {file_path}")
