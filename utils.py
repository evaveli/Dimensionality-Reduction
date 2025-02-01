import os
import pandas as pd
import time
from scipy.io import arff
import numpy as np


def load_datasets(directory, dataset_name):
    dataset_file = os.path.join(directory, f"{dataset_name}.arff")
    data, _ = arff.loadarff(dataset_file)
    return data


def get_process_time(process, start_time):
    """
    Calculates and logs the elapsed time for a given process.

    Args:
        process (str): Description of the process.
        start_time (float): Timestamp when the process started.

    Returns:
        float: Current timestamp after the process completion.

    """
    current_time = time.time()
    process_time = current_time - start_time
    print(f"Finished {process} in {process_time:.3f} seconds.")
    return current_time


def create_directory(directory):
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Directory path.

    Returns:
        output_dir (str): Full path to the output directory

    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_dir = directory
    return output_dir


def save_csv(X, name):
    np.savetxt(f"csv/{name}.csv", X, delimiter=",", comments="")


def check_missing_values(df):
    missing_values = []
    for column in df.columns:
        if (
            df[column].any() == None
            or df[column].any() == ""
            or df[column].isnull().sum() > 0
        ):
            missing_values.append(f"Missing values for column {column}.")

    if missing_values:
        [print(col) for col in missing_values]
        return True
    else:
        print("We are not missing any values!")
        return False


def save_metrics_to_csv(results, dataset_name, output_dir, k_alorithm_flag):
    """
    Save the clustering metrics to separate CSV files for each metric.
    """

    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    rows = []

    if k_alorithm_flag:  # K-Means
        for metric, k_values in results.items():
            for k, metrics in k_values.items():
                row = {"K": k, "distance_metric": metric}
                row.update(metrics)  # Add all metrics for this K
                rows.append(row)
    else:  # G-Means
        for metric, alphas in results.items():
            for alpha, metrics in alphas.items():
                row = {"Alpha": alpha, "distance_metric": metric}
                row.update(metrics)
                rows.append(row)

    # Convert rows to a DataFrame
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    csv_file = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
    df.to_csv(csv_file, index=False)
    print(f"Metrics for {metric} saved to {csv_file}")


def print_divider(char="=", length=60, before="", after=""):
    if before:
        print(before)
    print(char * length)
    if after:
        print(after)


def print_heading1(text):
    print("\n" + "=" * len(text))
    print(text.upper())
    print("=" * len(text) + "\n")


def print_heading2(text):
    print("\n" + text)
    print("-" * len(text) + "\n")


def print_heading3(text):
    print("\n" + text)
    print("".join(["-" if i % 2 == 0 else " " for i in range(len(text))]) + "\n")
