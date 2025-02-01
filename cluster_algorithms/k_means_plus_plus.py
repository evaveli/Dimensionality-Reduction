import numpy as np
from metrics.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)


class KMeansPlusPlus:
    def __init__(self, n_clusters, d_metric):
        self.n_clusters = n_clusters
        self.d_metric = d_metric
        self.centers = []

    def _find_closest_centroid(self, point):
        """
        Find the minimum distance between a point and all centroids.
        """
        min_distance = np.inf

        for center in self.centers:
            if self.d_metric == "euclidean":
                distance = euclidean_distance(point, center)
            elif self.d_metric == "manhattan":
                distance = manhattan_distance(point, center)
            elif self.d_metric == "cosine":
                distance = cosine_distance(point, center)

            # Assert new min distance
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def _select_weighted_centroid(self, distances):
        """
        Selects a new centroid based on the weighted probability of distances.
        """

        distances_squared = [distance**2 for distance in distances]
        distances_squared_sum = np.sum(distances_squared)
        probabilities = distances_squared / distances_squared_sum
        return np.random.choice(len(probabilities), p=probabilities.flatten())

    def get_controids(self, data, n_samples, random_state=None):
        """
        Identifies and returns the centroids based on the K-Means++ algorithm.
        """
        np.random.seed(random_state)

        # Step 1: Choose the first centre randomly
        first_center_idx = np.random.choice(n_samples)
        self.centers.append(data[first_center_idx])

        # Step 2: Choose subsequent centres with D**2 weighting
        for _ in range(1, self.n_clusters):
            distances = []

            for point in data:
                distances.append(self._find_closest_centroid(point))

            next_center_idx = self._select_weighted_centroid(distances)
            self.centers.append(data[next_center_idx])

        return np.array(self.centers)
