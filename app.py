import os
import time
import sys
import umap
import numpy as np
import pandas as pd
from preprocessing import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import OPTICS
from cluster_algorithms import optics
from cluster_algorithms.k_means import KMeans
from summary.summary_optics import OPTICSAnalyzer
from summary.summary_kmeans import KMeansAnalyzer
from pca.pca import ourpca, compare_arrays_elementwise
from pca.pca_visualization import PCAVisualizer
from utils import load_datasets


def display_menu():
    print("\nClustering Algorithm Selection:")
    print("1) PCA related menus")
    print("2) OPTICS algorithm")
    print("3) K-means++")
    print("4) Generate LateX Code/ Confusion matrix for Optics | K-Means++")
    print("5) Create 3D scatter plot of best models using the best 3 components")
    print("6) Compare K-means++ Clustering: Best Original Features vs PCA-reduced Data")
    print("7) Compare K-means++ Clustering: Best Original Features vs UMAP Data")
    print("8) Compare Optics Clustering: Best Original Features vs PCA-reduced Data")
    print("9) Compare Optics Clustering: Best Original Features vs UMAP Data")
    print("10) Compare Optics | K-means ++ Results vs Original Features ")
    print("11) Exit")
    return input("\nEnter your choice (1-11): ")


def display_menu_dataset_choice():
    print("Please select one of these options:")
    print("1) I would like to use the MinMax Scaled dataset")
    print("2) I would like to use the Robust Scaled dataset")
    print("3) I would like to use the Standard Scaled dataset")
    return input("\nEnter your choice (1-3): ")


def display_pca_menu():
    print("\nPCA Analysis Menu:")
    print("-" * 30)
    print("1) Validate our PCA against scikit-learn, show eigenvalues and vectors")
    print("2) Generate variance analysis and create PCA projections")
    print("3) Plot cumulative explained variance")
    print("4) Return to main menu")
    print("-" * 30)
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            return choice
        print("Invalid choice. Please select 1-4.")


def display_scatter_plot_menu():
    print("\n3D Scatter Plot Options:")
    print("-" * 30)
    print("1) Plot with PCA transformed data")
    print("2) Plot with original features")
    print("3) Return to PCA menu")
    print("-" * 30)

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please select 1-3.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "datasets")

    try:
        dataset_name = (
            input("Enter the name of the dataset ['cmc', 'pen-based']: ") or "cmc"
        )
        if dataset_name not in ["cmc", "pen-based"]:
            raise ValueError(f"Dataset must be one of: 'cmc',  'pen-based'")

    except ValueError as e:
        print(f"You might have misspelled the dataset name: {e}")
        exit(1)

    start_time = time.time()

    preprocessor = preprocessing.Preprocessing()

    # Load dataset
    df = pd.DataFrame(load_datasets(dataset_dir, dataset_name))

    # Preprocess the data
    X, X_robust, X_standard, y_true = preprocessor.generous_preprocessing(df)

    def get_selected_dataset(X, X_robust, X_standard):

        choice = display_menu_dataset_choice()

        if choice == "1":
            return X, "MinMax Scaled"
        elif choice == "2":
            return X_robust, "Robust Scaled"
        else:  # choice == '3'
            return X_standard, "Standard Scaled"

    binary_vars = preprocessor.binary_vars
    categorical_vars = preprocessor.categorical_vars
    while True:
        choice = display_menu()

        choice = int(choice)
        if choice == 11:
            print("Exiting the program...")
            sys.exit()

        elif choice == 1:  # PCA menu
            while True:
                pca_choice = display_pca_menu()
                pca_choice = int(pca_choice)

                if pca_choice == 4:
                    break  # Return to main menu

                # Get dataset choice
                X_to_use, dataset_type = get_selected_dataset(X, X_robust, X_standard)

                if pca_choice == 1:
                    # Validation against scikit-learn

                    (
                        transformed_our,
                        eigenvalues_our,
                        eigenvectors_our,
                        cov_matrix,
                        reconstructed_data,
                    ) = ourpca(X_to_use, n_components=3)
                    pca_sklearn = PCA(n_components=3)
                    transformed_sklearn = pca_sklearn.fit_transform(X_to_use)

                    ipca_sklearn = IncrementalPCA(n_components=3)
                    transformed_sklearn_ipca = ipca_sklearn.fit_transform(X_to_use)

                    print("\n=== PCA Analysis Details ===")
                    print("\nCovariance Matrix:")
                    print("-" * 50)
                    print(np.round(cov_matrix, 4))

                    print("\nEigenvalues:")
                    print("-" * 50)
                    for i, val in enumerate(eigenvalues_our):
                        print(f"λ{i+1}: {val:.4f}")

                    print("\nEigenvectors (columns represent principal components):")
                    print("-" * 50)
                    print(np.round(eigenvectors_our, 4))

                    print(
                        f"\nComparing our PCA vs sklearn PCAs results for {dataset_type} data:"
                    )
                    compare_arrays_elementwise(
                        transformed_our, transformed_sklearn, transformed_sklearn_ipca
                    )

                elif pca_choice == 2:

                    pca_viz = PCAVisualizer()
                    # Show variance analysis
                    pca = PCA()
                    pca.fit(X_to_use)
                    print(f"\nExplained Variance Analysis for {dataset_type} data:")
                    print("-" * 50)
                    print("Component    Explained Variance (%)    Cumulative (%)")
                    print("-" * 50)
                    cumsum = 0
                    for i, ratio in enumerate(pca.explained_variance_ratio_):
                        cumsum += ratio * 100
                        print(f"{i+1:^9d}    {ratio*100:^20.2f}    {cumsum:^14.2f}")

                    latex_table = pca_viz.print_and_generate_latex_variance_analysis(
                        X_to_use, dataset_name, scaling=dataset_type
                    )
                    print(
                        f"\nLaTeX table has been saved to: summary/Latex_tables/pca_{dataset_name}_{dataset_type}_variance.txt"
                    )

                    pca_viz.save_pca_projections(X_to_use, dataset_name, dataset_type)

                elif pca_choice == 3:
                    # Plot cumulative variance
                    pca_viz = PCAVisualizer()
                    title = f"Explained Variance vs Number of Components\n({dataset_type} Data)"
                    pca_viz.plot_explained_variance(
                        X_to_use, dataset_name, scaling_type=dataset_type, title=title
                    )

        elif choice == 2:  # optics
            thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
            dataset_choice = display_menu_dataset_choice()
            dataset_choice = int(dataset_choice)

            # Get scaling type for file naming
            if dataset_choice == 1:
                scaling_type = "MinMax Scaled"
            elif dataset_choice == 2:
                scaling_type = "Robust Scaled"
            else:
                scaling_type = "Standard Scaled"

            # For each threshold, load and process the corresponding PCA dataset
            for threshold in thresholds:
                print(f"\nProcessing PCA dataset with threshold {threshold}")

                # Load PCA-transformed dataset
                pca_file = f"pca/pca_projections/{dataset_name}_{scaling_type}_threshold_{str(threshold).replace('.','_')}.csv"

                try:
                    pca_data = pd.read_csv(pca_file)
                    X_pca = pca_data.values  # Convert to numpy array

                    # Initialize OPTICS with PCA threshold info
                    optics_clustering = optics.Optics(pca_threshold=threshold)

                    # Run OPTICS on the PCA-transformed data
                    optics_clustering.perform_optics(
                        dataset_name=dataset_name,
                        dataset_choice=dataset_choice,
                        X=X_pca,
                        y_true=y_true,
                    )

                except FileNotFoundError:
                    print(f"PCA transformed dataset not found: {pca_file}")
                    continue

        elif choice == 3:  # K-means++
            print("Running K-means++ on PCA transformed data...")
            perform_kmeans = KMeans()
            clustering_algo = "K-Means++"

            # Select correct scaling based on dataset
            scaling_type = "Robust Scaled" if dataset_name == "cmc" else "MinMax Scaled"

            # Process each threshold
            thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
            pca_dir = "pca/pca_projections"

            for threshold in thresholds:
                threshold_str = str(threshold).replace(".", "_")
                pca_file = f"{pca_dir}/{dataset_name}_{scaling_type}_threshold_{threshold_str}.csv"

                try:
                    print(f"\nProcessing PCA threshold: {threshold}")
                    pca_data = pd.read_csv(pca_file)
                    X_pca = pca_data.values

                    results = perform_kmeans.run_k_means_experiments(
                        X_pca,
                        dataset_choice=1,  # This might need adjustment in the KMeans class
                        dataset_name=dataset_name,
                        clustering_algo=f"{clustering_algo}_PCA_{threshold_str}",
                        y_true=y_true,
                        plus_plus=True,
                        threshold=threshold,  # Add the threshold parameter here
                    )

                except FileNotFoundError:
                    print(f"PCA transformed dataset not found: {pca_file}")
                    continue

            # End of K ++
        elif choice == 4:
            algorithm = input("Choose clustering algorithm (1: OPTICS, 2: K-Means): ")
            if algorithm not in ["1", "2"]:
                print("Invalid algorithm choice")
                continue

            thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
            scaling_type = "Robust Scaled" if dataset_name == "cmc" else "MinMax Scaled"
            pca_dir = "pca/pca_projections"

            if algorithm == "1":
                print("Creating LaTeX code and confusion matrix for OPTICS...")
                for threshold in thresholds:
                    pca_file = f"{pca_dir}/{dataset_name}_{scaling_type}_threshold_{str(threshold).replace('.','_')}.csv"
                    try:
                        pca_data = pd.read_csv(pca_file)
                        X_pca = pca_data.values
                        analyzer = OPTICSAnalyzer(true_labels=y_true, features=X_pca)
                        results, latex = analyzer.analyze_dataset(
                            dataset_name, external_weight=0.5
                        )
                        if results is not None:
                            print(
                                "LaTeX table and confusion matrix for OPTICS saved successfully."
                            )
                    except FileNotFoundError:
                        print(f"Warning: File not found: {pca_file}")

            elif algorithm == "2":
                print("Creating LaTeX code and confusion matrix for K-Means...")
                for threshold in thresholds:
                    pca_file = f"{pca_dir}/{dataset_name}_{scaling_type}_threshold_{str(threshold).replace('.','_')}.csv"
                    try:
                        pca_data = pd.read_csv(pca_file)
                        X_pca = pca_data.values
                        analyzer = KMeansAnalyzer(true_labels=y_true, features=X_pca)
                        results, latex = analyzer.analyze_dataset(
                            dataset_name, external_weight=0.5
                        )
                        if results is not None:
                            print(
                                "LaTeX table and confusion matrix for K-Means saved successfully."
                            )
                    except FileNotFoundError:
                        print(f"Warning: File not found: {pca_file}")

        elif choice == 5:
            if dataset_name == "cmc":
                X_to_use = X_robust
            else:
                X_to_use = X

            print("Creating triple comparison plot...")

            # Create visualization with all the computations included
            pca_viz = PCAVisualizer()
            pca_viz.plot_triple_clustering_comparison(
                X_data=X_to_use,
                dataset_name=dataset_name,
                external_weight=0.5,
                y_true=y_true,
            )

            pca_viz.plot_triple_clustering_comparison_umap(
                X_data=X_to_use,
                dataset_name=dataset_name,
                external_weight=0.5,
                y_true=y_true,
            )

        elif choice == 6:
            print("Creating comparison plots...")

            if dataset_name == "cmc":
                X_to_use = X_robust
                target_k = 3
                dataset_type = "Robust Scaled"

            else:
                X_to_use = X
                target_k = 10
                dataset_type = "MinMax Scaled"

            external_weight = 0.5

            # First do the PCA transformation with best threshold from config
            kmeans_config = pd.read_csv(
                f"summary/best_configs/best_kmeans_{dataset_name}_k{target_k}_w{external_weight}.csv"
            )
            threshold = kmeans_config["threshold"].iloc[0]
            transformed_data, _, eigenvectors, _, _ = ourpca(X_to_use, n_components=3)

            # Run K-means on both original and PCA data
            kmeans = KMeans()

            # For original data (best 3 features)
            total_contributions = np.sum(np.abs(eigenvectors[:, :3]), axis=1)
            top_features_idx = np.argsort(total_contributions)[::-1][:3]
            _, kmeans_labels_orig, _ = kmeans.k_means(
                X=X_to_use[:, top_features_idx],
                K=target_k,
                distance_metric="cosine",
                plus_plus=True,
                max_iters=100,
                random_state=42,
            )

            # For PCA data
            _, kmeans_labels_pca, _ = kmeans.k_means(
                X=transformed_data,
                K=target_k,
                distance_metric="cosine",
                plus_plus=True,
                max_iters=100,
                random_state=42,
            )

            # Create visualization using existing function
            pca_viz = PCAVisualizer()
            pca_viz.plot_3d_scatter_comparison(
                X_orig=X_to_use,
                X_pca=transformed_data,
                y=kmeans_labels_orig,
                y2=kmeans_labels_orig,
                orig_indices=top_features_idx,
                dataset_name=dataset_name,
                scaling_type=dataset_type,
                name="K Means",
                add_label="K-Means++ clustering",
            )
        elif choice == 8:
            print("Creating comparison plots...")

            if dataset_name == "cmc":
                X_to_use = X
                target_k = 3
                dataset_type = "MinMax Scaled"

                optics_orig = OPTICS(
                    min_samples=55, max_eps=1.5, metric="manhattan", algorithm="auto"
                )
                optics_pca = OPTICS(
                    min_samples=55, max_eps=2.2, metric="manhattan", algorithm="auto"
                )
            else:
                X_to_use = X
                target_k = 10
                dataset_type = "MinMax Scaled"

                optics_orig = OPTICS(
                    min_samples=20, max_eps=1, metric="cosine", algorithm="auto"
                )

                optics_pca = OPTICS(
                    min_samples=22,
                    max_eps=4.2,
                    metric="cosine",
                    algorithm="auto",
                    xi=0.047,
                )

            external_weight = 0.5

            # Load OPTICS config and get threshold
            optics_config = pd.read_csv(
                f"summary/best_configs/best_optics_{dataset_name}_k{target_k}_w{external_weight}.csv"
            )
            threshold = optics_config["threshold"].iloc[0]

            # PCA transformation
            transformed_data, _, eigenvectors, _, _ = ourpca(
                X_to_use, threshold=threshold
            )

            feature_vars = np.var(X_to_use, axis=0)
            top_features_idx = np.argsort(feature_vars)[::-1][:3]
            # For original data (best 3 features)
            # total_contributions = np.sum(np.abs(eigenvectors[:, :3]), axis=1)
            # top_features_idx = np.argsort(total_contributions)[::-1][:3]

            optics_labels_orig = optics_orig.fit_predict(X_to_use)
            optics_labels_pca = optics_pca.fit_predict(transformed_data)

            # Create visualization using existing function
            pca_viz = PCAVisualizer()
            pca_viz.plot_3d_scatter_comparison(
                X_orig=X_to_use,
                X_pca=transformed_data,
                y=optics_labels_orig,
                y2=optics_labels_pca,
                orig_indices=top_features_idx,
                dataset_name=dataset_name,
                scaling_type=dataset_type,
                name="OPTICS",
                add_label="OPTICS clustering",
            )

        elif choice == 9:
            print("Creating OPTICS vs UMAP comparison...")

            if dataset_name == "cmc":
                X_to_use = X
                target_k = 3
                dataset_type = "MinMax Scaled"
                optics_orig = OPTICS(
                    min_samples=55, max_eps=1.5, metric="manhattan", algorithm="auto"
                )
                optics_umap = OPTICS(
                    min_samples=45, max_eps=1.2, metric="manhattan", algorithm="auto"
                )
            else:
                X_to_use = X
                target_k = 10
                dataset_type = "MinMax Scaled"
                optics_orig = OPTICS(
                    min_samples=20, max_eps=3.5, metric="cosine", algorithm="auto"
                )
                optics_umap = OPTICS(
                    min_samples=145,
                    max_eps=0.35,
                    metric="cosine",
                    algorithm="auto",
                    xi=0.0999999,
                )
            # OPTICS clustering on original data

            # UMAP transformation
            umap_reducer = umap.UMAP(n_components=3, n_jobs=-1)
            umap_embedded = umap_reducer.fit_transform(X_to_use)
            feature_vars = np.var(X_to_use, axis=0)
            top_features_idx = np.argsort(feature_vars)[::-1][:3]

            optics_labels = optics_orig.fit_predict(X_to_use)
            optics_labels_umap = optics_umap.fit_predict(umap_embedded)
            # Create visualization using modified plotting function
            pca_viz = PCAVisualizer()
            pca_viz.plot_3d_scatter_comparison(
                X_orig=X_to_use,
                X_pca=umap_embedded,
                y=optics_labels,
                y2=optics_labels_umap,
                orig_indices=top_features_idx,
                dataset_name=dataset_name,
                scaling_type=dataset_type,
                name="OPTICS_UMAP",
                add_label="OPTICS clustering",
            )

        elif choice == 7:
            print("Creating UMAP vs K-means comparison plots...")

            if dataset_name == "cmc":
                X_to_use = X_robust
                target_k = 3
                dataset_type = "Robust Scaled"
            else:
                X_to_use = X
                target_k = 10
                dataset_type = "MinMax Scaled"

            external_weight = 0.5

            # Load K-means config
            kmeans_config = pd.read_csv(
                f"summary/best_configs/best_kmeans_{dataset_name}_k{target_k}_w{external_weight}.csv"
            )

            # UMAP transformation
            umap_reducer = umap.UMAP(n_components=3, random_state=42)
            umap_embedded = umap_reducer.fit_transform(X_to_use)

            # Feature selection for original data
            feature_vars = np.var(X_to_use, axis=0)
            top_features_idx = np.argsort(feature_vars)[::-1][:3]

            # Run K-means on both original and UMAP data
            kmeans = KMeans()

            # For original data
            _, kmeans_labels_orig, _ = kmeans.k_means(
                X=X_to_use[:, top_features_idx],
                K=target_k,
                distance_metric="cosine",
                plus_plus=True,
                max_iters=100,
                random_state=42,
            )

            # For UMAP data
            _, kmeans_labels_umap, _ = kmeans.k_means(
                X=umap_embedded,
                K=target_k,
                distance_metric="cosine",
                plus_plus=True,
                max_iters=100,
                random_state=42,
            )

            # Create visualization
            pca_viz = PCAVisualizer()
            pca_viz.plot_3d_scatter_comparison(
                X_orig=X_to_use,
                X_pca=umap_embedded,
                y=kmeans_labels_orig,
                y2=kmeans_labels_umap,  # Using UMAP K-means results
                orig_indices=top_features_idx,
                dataset_name=dataset_name,
                scaling_type=dataset_type,
                name="KMeans_UMAP",
                add_label="K-Means++ clustering comparison",
            )
        elif choice == 10:
            print("Creating K-means vs OPTICS comparison on original data...")

            if dataset_name == "cmc":
                X_to_use = X
                target_k = 5  # K-means target
                dataset_type = "MinMax Scaled"

                # OPTICS configuration for CMC
                optics_orig = OPTICS(
                    min_samples=55,
                    max_eps=1.5,
                    metric="manhattan",
                    algorithm="auto",
                    n_jobs=-1,
                )

            else:  # pen-based dataset
                X_to_use = X
                target_k = 12  # K-means target
                dataset_type = "MinMax Scaled"

                # OPTICS configuration for pen-based
                optics_orig = OPTICS(
                    min_samples=20,
                    max_eps=1,
                    metric="cosine",
                    algorithm="auto",
                    n_jobs=-1,
                )

            # Apply K-means
            kmeans = KMeans()
            _, kmeans_labels, _ = kmeans.k_means(
                X=X_to_use,
                K=target_k,
                distance_metric="cosine",
                max_iters=100,
                plus_plus=True,
                random_state=42,
            )

            # Apply OPTICS
            optics_labels = optics_orig.fit_predict(X_to_use)

            # Select top 3 features based on variance
            feature_vars = np.var(X_to_use, axis=0)
            top_features_idx = np.argsort(feature_vars)[::-1][:3]

            # Create visualization using modified plotting function
            pca_viz = PCAVisualizer()
            pca_viz.plot_kmeans_optics_comparison(
                X_data=X_to_use,
                kmeans_labels=kmeans_labels,
                optics_labels=optics_labels,
                feature_indices=top_features_idx,
                dataset_name=dataset_name,
                scaling_type=dataset_type,
            )

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
