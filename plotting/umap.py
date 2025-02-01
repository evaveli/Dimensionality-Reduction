import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
import os

class UMAPVisualizer:
    def __init__(self, output_path="plots/umap/"):
        """
        Initialize UMAP visualizer
        
        Parameters:
        output_path (str): Directory to save plots
        """
        # Convert to absolute path
        self.output_path = os.path.abspath(output_path)
        
        # Create directory if it doesn't exist (using exist_ok=True)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def _hungarian_matching(self, y_true, y_pred):
        """
        Match cluster labels to true labels using Hungarian algorithm
        """
        # Create contingency matrix
        classes = np.unique(y_true)
        clusters = np.unique(y_pred)
        contingency = np.zeros((len(clusters), len(classes)))
        
        for i, cluster in enumerate(clusters):
            for j, cls in enumerate(classes):
                contingency[i, j] = np.sum((y_pred == cluster) & (y_true == cls))
                
        # Use Hungarian algorithm to find optimal matching
        row_ind, col_ind = linear_sum_assignment(-contingency)
        
        # Create mapping from cluster labels to true labels
        mapping = dict(zip(clusters, classes[col_ind]))
        
        # Apply mapping to predicted labels
        y_pred_matched = np.array([mapping[label] for label in y_pred])
        
        return y_pred_matched
        
    def fit_and_plot(self, X, y_true, clustering_params):
        """
        Fit Spectral Clustering, create UMAP projection, and plot comparison
        """
        # Create UMAP projection
        umap = UMAP(n_components=2, random_state=42)
        X_umap = umap.fit_transform(X)
        
        # Perform spectral clustering
        spectral = SpectralClustering(
            n_clusters=clustering_params['n_clusters'],
            affinity=clustering_params['affinity'],
            n_neighbors=clustering_params['n_neighbors'],
            eigen_solver=clustering_params['eigen_solver'],
            n_init=clustering_params['n_init'],
            random_state=42
        )
        predicted_labels = spectral.fit_predict(X)
        
        # Match predicted labels to true labels
        predicted_labels_matched = self._hungarian_matching(y_true, predicted_labels)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot predicted clusters
        scatter1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=predicted_labels_matched, cmap='tab10')
        ax1.set_title('UMAP with Predicted Clusters')
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot true labels
        scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y_true, cmap='tab10')
        ax2.set_title('UMAP with True Labels')
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.suptitle('UMAP Projections: Predicted vs True Labels')
        plt.savefig(os.path.join(self.output_path, 'umap_comparison.png'))
        plt.close()