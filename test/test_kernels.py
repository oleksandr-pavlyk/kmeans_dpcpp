import pytest
import dpctl
import dpctl.tensor as dpt
import kmeans_dpcpp as kdp

import numpy as np

def test_broadcast_divide():
    X = dpt.full((16, 5), 27.0, dtype='f4')
    y = dpt.asarray([1, 3, 9, 3, 1], dtype='f4')

    ht, _ = kdp.broadcast_divide(X, y, sycl_queue=X.sycl_queue)
    ht.wait()

    X_np = dpt.asnumpy(X)
    X_expected = np.full((16, 5), np.array([27, 9, 3, 9, 27], dtype='f4'))
    tol = np.finfo(X.dtype).resolution
    assert np.allclose( X_np, X_expected, rtol=tol )

    
def test_half_l2_norm_squared():
    X1 = dpt.ones((3, 2), dtype='f4')
    X2 = dpt.full((3, 4), 2., dtype='f4')
    X = dpt.empty((3, 6), dtype='f4')
    X[:, :4] = X2
    X[:, 4:] = X1
    y = dpt.empty((X.shape[1],), dtype='f4')

    ht, _ = kdp.half_l2_norm_squared(X, y, sycl_queue=X.sycl_queue)
    ht.wait()

    y_np = dpt.asnumpy(y)
    y_expected = np.array([6., 6., 6., 6., 1.5, 1.5], dtype='f4')
    tol = np.finfo(X.dtype).resolution
    assert np.allclose( y_np, y_expected, rtol=tol )


def test_reduce_centroids_data():
    n_copies = 3
    n_features = 2
    n_clusters = 4

    dataT = np.dtype('f4')
    indT = np.dtype('i4')

    cluster_sizes_private_copies = dpt.asarray(
        [ [ 2, 1, 3, 1],
          [ 1, 2, 3, 1],
          [ 0, 1, 0, 1] ], dtype=dataT)
    Xnp = np.random.randn(n_copies, n_features, n_clusters).astype(dataT)
    centroids_t_private_copies = dpt.asarray(Xnp, dtype=dataT)

    out_cluster_sizes = dpt.empty(n_clusters, dtype=dataT)
    out_centroids_t = dpt.empty((n_features, n_clusters,), dtype=dataT)
    canary_v = 10
    out_empty_clusters_list = dpt.full((n_clusters,), canary_v, dtype=indT)
    out_n_empty_clusters = dpt.zeros((1,), dtype=indT)

    q = cluster_sizes_private_copies.sycl_queue
    ht, _, = kdp.reduce_centroids_data(
        cluster_sizes_private_copies,  # (n_copies, n_clusters)
        centroids_t_private_copies,    # (n_copies, n_features, n_clusters,)
        out_cluster_sizes,             # (n_clusters, )
        out_centroids_t,               # (n_features, n_clusters,)
        out_empty_clusters_list,       # (n_clusters,)
        out_n_empty_clusters,          # (1,)
        sycl_queue=q
    )
    ht.wait()

    assert int(out_n_empty_clusters) == 0
    assert np.array_equal(dpt.asnumpy(out_empty_clusters_list), np.full(n_clusters, canary_v, dtype=indT))
    assert np.allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0), rtol = np.finfo(dataT).resolution)
    assert np.allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0))

    
def test_reduce_centroids_data_empty():
    n_copies = 3
    n_features = 2
    n_clusters = 4

    dataT = np.dtype('f4')
    indT = np.dtype('i4')

    cluster_sizes_private_copies = dpt.asarray(
        [ [ 2, 1, 3, 0],
          [ 1, 2, 3, 0],
          [ 1, 1, 0, 0] ], dtype=dataT)
    Xnp = np.random.randn(n_copies, n_features, n_clusters).astype(dataT)
    np.put(Xnp, np.arange(3, 12, step=4, dtype=np.int32), np.zeros(n_copies, dtype=dataT))
    centroids_t_private_copies = dpt.asarray(Xnp, dtype=dataT)

    out_cluster_sizes = dpt.empty(n_clusters, dtype=dataT)
    out_centroids_t = dpt.empty((n_features, n_clusters,), dtype=dataT)
    canary_v = 10
    out_empty_clusters_list = dpt.full((n_clusters,), canary_v, dtype=indT)
    out_n_empty_clusters = dpt.zeros((1,), dtype=indT)

    q = cluster_sizes_private_copies.sycl_queue
    ht, _, = kdp.reduce_centroids_data(
        cluster_sizes_private_copies,  # (n_copies, n_clusters)
        centroids_t_private_copies,    # (n_copies, n_features, n_clusters,)
        out_cluster_sizes,             # (n_clusters, )
        out_centroids_t,               # (n_features, n_clusters,)
        out_empty_clusters_list,       # (n_clusters,)
        out_n_empty_clusters,          # (1,)
        sycl_queue=q
    )
    ht.wait()

    assert int(out_n_empty_clusters) == 1
    assert np.array_equal(dpt.asnumpy(out_empty_clusters_list), np.array([3,canary_v,canary_v,canary_v], dtype=indT))
    assert np.allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0), rtol = np.finfo(dataT).resolution)
    assert np.allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0))


def test_compute_threshold():
    dataT = dpt.float32
    n = 10**5
    Xnp = np.random.randn(n).astype(dataT)
    data = dpt.asarray(Xnp, dtype=dataT)

    threshold = dpt.empty(tuple(), dtype=dataT)

    q = data.sycl_queue
    ht, _ = kdp.compute_threshold(data, 1, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(max(Xnp))

    ht, _ = kdp.compute_threshold(data, 2, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(np.partition(Xnp, kth = n - 2)[n-2])


def test_select_samples_far_from_centroid_kernel():
    dataT = dpt.float32
    indT = dpt.int32
    n = 10**5
    n_empty_clusters = 3
    Xnp = np.random.standard_gamma(4, size=n).astype(dataT)

    distance_to_centroid = dpt.asarray(Xnp, dtype=dataT)
    threshold = dpt.empty(tuple(), dtype=dataT)

    selected_samples_idx = dpt.full(n, 172, dtype=indT)
    n_selected_gt_threshold = dpt.zeros(tuple(), dtype=indT)
    n_selected_eq_threshold = dpt.ones(tuple(), dtype=indT)

    q = threshold.sycl_queue
    ht_ev, c_ev = kdp.compute_threshold(
        distance_to_centroid, n_empty_clusters, threshold, sycl_queue=q)

    # puts indices of distances greater than threshold at the beginning of selected_samples_idx
    # and indices of distances equal to threshold at the end of the selected_samples_idx
    ht_ev2, _ = kdp.select_samples_far_from_centroid(
        n_empty_clusters, distance_to_centroid, threshold,
        selected_samples_idx, n_selected_gt_threshold, n_selected_eq_threshold, 
        sycl_queue=q,
        depends=[c_ev]
    )

    ht_ev2.wait()
    ht_ev.wait()

    assert int(n_selected_gt_threshold) + int(n_selected_eq_threshold) - 1 == n_empty_clusters
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[:int(n_selected_gt_threshold)])] > float(threshold))
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[1-int(n_selected_eq_threshold):])] == float(threshold))


def test_relocate_empty_cluster():
    dataT = np.float32
    indT = np.uint32
    n = 8

    # 2 clusters, 8 3D points, uniform samples
    Xnp = np.array([
            [+1, +1, +1],
            [+1, +1, -1],
            [+1, -1, +1],
            [-1, +1, +1],
            [-1, -1, +1],
            [-1, +1, -1],
            [+1, -1, -1],
            [-1, -1, -1],
        ], dtype=dataT)
    Xnp_t = np.ascontiguousarray(Xnp.T)

    Cnp = np.array([[0.1, -0.1, 0.1], [5, 5, 5]], dtype=dataT)
    Cnp_t = np.ascontiguousarray(Cnp.T)


    sample_weights = dpt.ones(n, dtype=dataT)
    X_t = dpt.asarray(Xnp_t, dtype=dataT)
    centroid_t = dpt.asarray(Cnp_t, dtype=dataT)
    empty_clusters_list = dpt.asarray([1, 0], dtype=indT)
    sq_dist_to_nearest_centroid_np = np.min(np.square(Xnp[:, np.newaxis, :] - Cnp[np.newaxis, :, :]).sum(axis=-1), axis=-1)
    sq_dist_to_nearest_centroid = dpt.asarray(sq_dist_to_nearest_centroid_np, dtype=dataT)

    cluster_sizes = dpt.asarray([8, 0], dtype=dataT)
    per_sample_inertia = sq_dist_to_nearest_centroid
    n_empty_clusters = 1

    assignment_id = dpt.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=indT)

    q = X_t.sycl_queue
    ht, _ = kdp.relocate_empty_clusters(
        n_empty_clusters,
        X_t, sample_weights, assignment_id, empty_clusters_list, sq_dist_to_nearest_centroid,
        centroid_t, cluster_sizes, per_sample_inertia,
        sycl_queue = q
    )
    ht.wait()

    # centroid_t, cluster_sizes, per_sample_ineria change
    expected_updated_centroid_t = np.array([[0.1, -1], [-0.1, -1], [0.1, -1]], dtype=dataT)
    expected_cluster_sizes = np.array([7, 1], dtype=dataT)
    expected_per_sample_inertia = np.zeros(n, dtype=dataT)

    assert np.allclose(expected_updated_centroid_t, dpt.asnumpy(centroid_t), rtol=np.finfo(dataT).resolution)
    assert np.allclose(expected_cluster_sizes, dpt.asnumpy(cluster_sizes), rtol=np.finfo(dataT).resolution)
    assert np.allclose(expected_per_sample_inertia, dpt.asnumpy(per_sample_inertia), rtol=np.finfo(dataT).resolution)