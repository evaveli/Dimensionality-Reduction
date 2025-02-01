import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils import create_directory


def plot_confusion(cm_mapped, K, metric, dataset_name, scaling_method, algo):
    # Create dir if not existing
    plot_dir = os.path.join("plots", "confusion_matrices")
    create_directory(plot_dir)

    file_path = os.path.join(
        plot_dir,
        f"conf_m_{dataset_name}_{scaling_method}_{algo}_{metric}",
    )

    # Output confusion matrix to a file
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_mapped, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        f"Confusion Matrix for {dataset_name} using {algo} with K={K}, metric='{metric}'"
    )
    plt.savefig(file_path)
    plt.close()
