from ._kmeans_dpcpp import (
    broadcast_divide,
    half_l2_norm_squared,
    reduce_centroids_data,
    compute_threshold,
    select_samples_far_from_centroid,
    relocate_empty_clusters,
    compute_centroid_shifts_squared,
    compute_centroid_to_sample_distances,
    assignment,
    compute_inertia,
    reduce_vector_blocking,
    fused_lloyd_single_step,
    compute_number_of_private_copies,
)

__all__ = [
    "broadcast_divide",
    "half_l2_norm_squared",
    "reduce_centroids_data",
    "compute_threshold",
    "select_samples_far_from_centroid",
    "relocate_empty_clusters",
    "compute_centroid_shifts_squared",
    "compute_centroid_to_sample_distances",
    "assignment",
    "compute_inertia",
    "reduce_vector_blocking",
    "fused_lloyd_single_step",
    "compute_number_of_private_copies",
]

__doc__ = """
This module implements DPC++ offloading routines necessary for implementing Lloyd's algorithm to solve K-Means problem.
"""