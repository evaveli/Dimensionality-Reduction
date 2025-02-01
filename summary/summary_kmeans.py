import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class KMeansAnalyzer:
    def __init__(
        self,
        base_path="results/k_means_results",
        true_labels=None,
        features=None,
        external_weight=0.5,
    ):
        self.base_path = base_path
        self.true_labels = true_labels
        self.features = features
        self.external_weight = external_weight

    def analyze_dataset(self, dataset_name, external_weight=None):
        if external_weight is not None:
            self.external_weight = external_weight

        thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

        # Generate file paths for K-Means results
        # Select correct scaling based on dataset

        if dataset_name == "cmc":
            files = [
                f"{self.base_path}/{dataset_name}_Robust_threshold_{str(t).replace('.', '_')}_K-Means++_PCA_{str(t).replace('.', '_')}_metrics.csv"
                for t in thresholds
            ]
        else:  # For other datasets like 'pen-based' or 'hepatitis'
            files = [
                f"{self.base_path}/{dataset_name}_MinMax_threshold_{str(t).replace('.', '_')}_K-Means++_PCA_{str(t).replace('.', '_')}_metrics.csv"
                for t in thresholds
            ]

        dataframes = []
        for f, threshold in zip(files, thresholds):
            try:
                df = pd.read_csv(f)
                df["threshold"] = threshold  # Add threshold directly to DataFrame
                dataframes.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found: {f}")
                continue

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)

        try:
            results = self._find_best_configurations(
                combined_df, dataset_name, external_weight=self.external_weight
            )
            if results is None:
                return None, None
            latex_table = self._generate_latex_table(
                results, dataset_name, external_weight
            )
            return results, latex_table
        except Exception as e:
            print(f"Error in analyze_dataset: {str(e)}")
            return None, None

    def _scale_metrics(self, df):
        try:
            scaler = MinMaxScaler()
            intra_metrics = [
                "silhouette_score",
                "davies_bouldin",
                "calinski_harabasz",
            ]
            extra_metrics = [
                "adjusted_rand_index",
                "purity_score",
                "fowlkes_mallows_index",
                "normalized_mutual_info",
            ]

            df_new = df.copy()
            df_new[["sil_scaled", "dbi_scaled", "ch_scaled"]] = scaler.fit_transform(
                df[intra_metrics]
            )
            df_new["dbi_scaled"] = 1 - df_new["dbi_scaled"]
            df_new[["ari_scaled", "pur_scaled", "fmi_scaled", "nmi_scaled"]] = (
                scaler.fit_transform(df[extra_metrics])
            )

            return df_new
        except Exception as e:
            print(f"Error in _scale_metrics: {str(e)}")
            return None

    def _find_best_configurations(
        self, df, dataset_name, additional_params=None, external_weight=0.5
    ):
        df_scaled = self._scale_metrics(df)
        if df_scaled is None:
            return None

        intra_scaled = ["sil_scaled", "dbi_scaled", "ch_scaled"]
        extra_scaled = ["ari_scaled", "pur_scaled", "fmi_scaled", "nmi_scaled"]

        internal_weight = 1 - external_weight
        df_scaled["weighted_score"] = internal_weight * (
            df_scaled[intra_scaled].sum(axis=1) / len(intra_scaled)
        ) + external_weight * (df_scaled[extra_scaled].sum(axis=1) / len(extra_scaled))

        true_n_clusters = len(set(self.true_labels))
        best_configs = df_scaled.loc[df_scaled.groupby("K")["weighted_score"].idxmax()]

        matching_configs = best_configs[best_configs["K"] == true_n_clusters]
        if not matching_configs.empty:
            best_config = matching_configs.iloc[0]
            self._generate_confusion_matrix(best_config, dataset_name, external_weight)

        return best_configs

    def _generate_confusion_matrix(self, best_config, dataset_name, external_weight):
        kmeans = KMeans(
            n_clusters=int(best_config["K"]),
            init="k-means++",
            max_iter=300,
            random_state=42,
        )

        labels = kmeans.fit_predict(self.features)
        mask = labels != -1
        labels = labels[mask]
        true_labels = self.true_labels[mask]

        cm = confusion_matrix(true_labels, labels)
        mask = ~(cm == 0).all(axis=1)
        cm = cm[mask][:, ~(cm == 0).all(axis=0)]

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        os.makedirs("plots/confusion_matrices", exist_ok=True)
        plt.title(
            f"Confusion Matrix for {dataset_name}\nBest Configuration (External Weight: {external_weight})"
        )
        plt.savefig(
            f"plots/confusion_matrices/conf_matrix_{dataset_name}_kmeans_w{external_weight}.png"
        )
        plt.close()

    def _generate_latex_table(self, df, dataset_name, external_weight):
        df = df.copy().reset_index(drop=True)

        table = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n"
            "\\hline\n"
            "k & threshold & sil & db & ch & ari & fmi & purity \\\\\n"
            "\\hline\n"
        )

        for idx, row in df.iterrows():
            line = [
                str(int(row["K"])),
                f"{row['threshold']:.2f}",  # Use the threshold we added
                f"{row['silhouette_score']:.3f}",
                f"{row['davies_bouldin']:.3f}",
                f"{row['calinski_harabasz']:.3f}",
                f"{row['adjusted_rand_index']:.3f}",
                f"{row['fowlkes_mallows_index']:.3f}",
                f"{row['purity_score']:.3f}",
            ]
            table += " & ".join(line) + " \\\\\n\\hline\n"

        table += (
            "\\end{tabular}}\n"
            f"\\caption{{Best {dataset_name} K-Means Clustering Results per k (External Weight: {external_weight:.1f})}}\n"
            f"\\label{{tab:{dataset_name}_kmeans}}\n"
            "\\end{table}"
        )

        os.makedirs("summary/Latex_tables", exist_ok=True)
        with open(
            f"summary/Latex_tables/best_{dataset_name}_kmeans_w{external_weight}.txt",
            "w",
        ) as f:
            f.write(table)
        if dataset_name == "cmc":
            target_k = 3
        elif dataset_name == "hepatitis":
            target_k = 2
        elif dataset_name == "pen-based":
            target_k = 10

        best_k10 = df[df["K"] == target_k]

        if not best_k10.empty:
            # Select relevant columns and save to CSV
            columns_to_save = ["K", "threshold"]

            csv_path = os.path.join("summary", "best_configs")
            os.makedirs(csv_path, exist_ok=True)

            best_k10[columns_to_save].to_csv(
                f"{csv_path}/best_kmeans_{dataset_name}_k{target_k}_w{external_weight}.csv",
                index=False,
            )
            print(f"Best configuration for k={target_k} saved to CSV")

        return table
