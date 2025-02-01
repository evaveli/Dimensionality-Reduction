import numpy as np


def minkowski_distance(p, points, reference_point):
    """
    Calculate the Minkowski distance between a set of points or a single point and a reference point.

    Parameters:
    - p: The Minkowski order (e.g., p=2 for Euclidean, p=1 for Manhattan).
    - points: The array of points or a single point to compare
    - reference_point: The reference point
    """
    summation_axis = 0 if len((points * reference_point).shape) == 1 else 1

    if p == np.inf:  # Handle special case for Chebyshev distance (p=∞)
        return np.max(abs(points - reference_point), axis=summation_axis)
    return np.sum((abs(points - reference_point)) ** p, axis=summation_axis) ** (1 / p)


def cosine_distance(points, reference_point):
    """
    Calculate the cosine distance between a set of points and a reference point.

    Parameters:
    - points: The array of points to compare or a single point to compare
    - reference_point: The reference point
    """
    summation_axis = 0 if len((points * reference_point).shape) == 1 else 1
    dot_product = np.sum(points * reference_point, axis=summation_axis)
    magnitude_product = np.sqrt(np.sum(points**2, axis=summation_axis)) * np.sqrt(
        np.sum(reference_point**2)
    )
    return 1 - (dot_product / magnitude_product if magnitude_product.any() != 0 else 0)


def manhattan_distance(points, reference_point):
    return minkowski_distance(
        1,
        points,
        reference_point,
    )


def euclidean_distance(points, reference_point):
    return minkowski_distance(
        2,
        points,
        reference_point,
    )
