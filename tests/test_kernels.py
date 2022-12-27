import pytest
import dpctl
import dpctl.tensor as dpt
import kmeans_dpcpp as kdp

from numpy.testing import assert_allclose, assert_array_equal

import numpy as np

def _no_fp64_support():
    return not dpctl.SyclDevice().has_aspect_fp64


def test_broadcast_divide_fp32():
    dataT = dpt.float32
    X = dpt.full((16, 5), 27.0, dtype=dataT)
    y = dpt.asarray([1, 3, 9, 3, 1], dtype=dataT)

    ht, _ = kdp.broadcast_divide(X, y,
        work_group_size=32, sycl_queue=X.sycl_queue)
    ht.wait()

    X_np = dpt.asnumpy(X)
    X_expected = np.full((16, 5), np.array([27, 9, 3, 9, 27], dtype=dataT))
    tol = np.finfo(X.dtype).resolution
    assert_allclose( X_np, X_expected, rtol=tol)


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_broadcast_divide_fp64():
    dataT = dpt.float64
    X = dpt.full((16, 5), 27.0, dtype=dataT)
    y = dpt.asarray([1, 3, 9, 3, 1], dtype=dataT)

    ht, _ = kdp.broadcast_divide(X, y,
        work_group_size=32, sycl_queue=X.sycl_queue)
    ht.wait()

    X_np = dpt.asnumpy(X)
    X_expected = np.full((16, 5), np.array([27, 9, 3, 9, 27], dtype=dataT))
    tol = np.finfo(X.dtype).resolution
    assert_allclose( X_np, X_expected, rtol=tol)


def test_half_l2_norm_squared_fp32():
    dataT = dpt.float32
    X1 = dpt.ones((3, 2), dtype=dataT)
    X2 = dpt.full((3, 4), 2., dtype=dataT)
    X = dpt.empty((3, 6), dtype=dataT)
    X[:, :4] = X2
    X[:, 4:] = X1
    y = dpt.empty((X.shape[1],), dtype=dataT)

    ht, _ = kdp.half_l2_norm_squared(X, y,
        work_group_size=32, sycl_queue=X.sycl_queue)
    ht.wait()

    y_np = dpt.asnumpy(y)
    y_expected = np.array([6., 6., 6., 6., 1.5, 1.5], dtype=dataT)
    tol = np.finfo(X.dtype).resolution
    assert_allclose( y_np, y_expected, rtol=tol)


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_half_l2_norm_squared_fp64():
    dataT = dpt.float64
    X1 = dpt.ones((3, 2), dtype=dataT)
    X2 = dpt.full((3, 4), 2., dtype=dataT)
    X = dpt.empty((3, 6), dtype=dataT)
    X[:, :4] = X2
    X[:, 4:] = X1
    y = dpt.empty((X.shape[1],), dtype=dataT)

    ht, _ = kdp.half_l2_norm_squared(X, y,
        work_group_size=32, sycl_queue=X.sycl_queue)
    ht.wait()

    y_np = dpt.asnumpy(y)
    y_expected = np.array([6., 6., 6., 6., 1.5, 1.5], dtype=dataT)
    tol = np.finfo(X.dtype).resolution
    assert_allclose( y_np, y_expected, rtol=tol)


def test_reduce_centroids_data_fp32():
    n_copies = 3
    n_features = 2
    n_clusters = 4

    dataT = np.float32
    indT = np.int32

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
    ht, ev = kdp.reduce_centroids_data(
        cluster_sizes_private_copies,  # (n_copies, n_clusters)
        centroids_t_private_copies,    # (n_copies, n_features, n_clusters,)
        out_cluster_sizes,             # (n_clusters, )
        out_centroids_t,               # (n_features, n_clusters,)
        out_empty_clusters_list,       # (n_clusters,)
        out_n_empty_clusters,          # (1,)
        work_group_size=32,
        sycl_queue=q
    )
    ht.wait()
    ev.wait()

    assert int(out_n_empty_clusters) == 0
    assert np.array_equal(dpt.asnumpy(out_empty_clusters_list), np.full(n_clusters, canary_v, dtype=indT))
    assert np.allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0),
        rtol = np.finfo(dataT).resolution)
    assert np.allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0),
        rtol = np.finfo(dataT).resolution)


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_reduce_centroids_data_fp64():
    n_copies = 3
    n_features = 2
    n_clusters = 4

    dataT = np.dtype('f8')
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
        work_group_size=256,
        sycl_queue=q
    )
    ht.wait()

    assert int(out_n_empty_clusters) == 0
    assert np.array_equal(dpt.asnumpy(out_empty_clusters_list), np.full(n_clusters, canary_v, dtype=indT))
    assert np.allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0),
        rtol = np.finfo(dataT).resolution)
    assert np.allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0),
        rtol = np.finfo(dataT).resolution)


def test_reduce_centroids_data_empty_fp32():
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
        work_group_size=128,
        sycl_queue=q
    )
    ht.wait()

    assert int(out_n_empty_clusters) == 1
    assert_array_equal(dpt.asnumpy(out_empty_clusters_list), np.array([3,canary_v,canary_v,canary_v], dtype=indT))
    assert_allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0), rtol = np.finfo(dataT).resolution)
    assert_allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0))


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_reduce_centroids_data_empty_fp64():
    n_copies = 3
    n_features = 2
    n_clusters = 4

    dataT = np.float64
    indT = np.int32

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
        work_group_size=128,
        sycl_queue=q
    )
    ht.wait()

    assert int(out_n_empty_clusters) == 1
    assert_array_equal(dpt.asnumpy(out_empty_clusters_list), np.array([3,canary_v,canary_v,canary_v], dtype=indT))
    assert_allclose(
        dpt.asnumpy(out_centroids_t), Xnp.sum(axis=0), rtol = np.finfo(dataT).resolution)
    assert_allclose(
        dpt.asnumpy(out_cluster_sizes), dpt.asnumpy(cluster_sizes_private_copies).sum(axis=0))


def test_compute_threshold_1_fp32():
    dataT = dpt.float32
    n = 255*23
    Xnp = np.random.randn(n).astype(dataT)
    data = dpt.asarray(Xnp, dtype=dataT)

    threshold = dpt.full(tuple(), -127, dtype=dataT)

    q = data.sycl_queue
    ht, _ = kdp.compute_threshold(data, 1, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(max(Xnp))


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_compute_threshold_1_fp64():
    dataT = dpt.float64
    n = 255*23
    Xnp = np.random.randn(n).astype(dataT)
    data = dpt.asarray(Xnp, dtype=dataT)

    threshold = dpt.full(tuple(), -127, dtype=dataT)

    q = data.sycl_queue
    ht, _ = kdp.compute_threshold(data, 1, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(max(Xnp))


def test_compute_threshold_2_fp32():
    dataT = dpt.float32
    n = 255*23
    Xnp = np.random.randn(n).astype(dataT)
    data = dpt.asarray(Xnp, dtype=dataT)

    threshold = dpt.full(tuple(), -127, dtype=dataT)

    q = data.sycl_queue
    ht, _ = kdp.compute_threshold(data, 2, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(np.partition(Xnp, kth = n - 2)[n-2])


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_compute_threshold_2_fp64():
    dataT = dpt.float64
    n = 255*23
    Xnp = np.random.randn(n).astype(dataT)
    data = dpt.asarray(Xnp, dtype=dataT)

    threshold = dpt.full(tuple(), -127, dtype=dataT)

    q = data.sycl_queue
    ht, _ = kdp.compute_threshold(data, 2, threshold, sycl_queue=q)
    ht.wait()

    assert float(threshold) == float(np.partition(Xnp, kth = n - 2)[n-2])


def test_select_samples_far_from_centroid_kernel_fp32():
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
        work_group_size=256,
        sycl_queue=q,
        depends=[c_ev]
    )

    ht_ev2.wait()
    ht_ev.wait()

    assert int(n_selected_gt_threshold) + int(n_selected_eq_threshold) - 1 == n_empty_clusters
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[:int(n_selected_gt_threshold)])] > float(threshold))
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[1-int(n_selected_eq_threshold):])] == float(threshold))


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_select_samples_far_from_centroid_kernel_fp64():
    dataT = dpt.float64
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
        work_group_size=256,
        sycl_queue=q,
        depends=[c_ev]
    )

    ht_ev2.wait()
    ht_ev.wait()

    assert int(n_selected_gt_threshold) + int(n_selected_eq_threshold) - 1 == n_empty_clusters
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[:int(n_selected_gt_threshold)])] > float(threshold))
    assert np.all(Xnp[dpt.asnumpy(selected_samples_idx[1-int(n_selected_eq_threshold):])] == float(threshold))


def test_relocate_empty_clusters_fp32():
    dataT = np.float32
    indT = np.int32
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

    Cnp = np.array([[0.11, -0.1, 0.09], [5, 5, 5]], dtype=dataT)
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
        work_group_size=256,
        sycl_queue = q
    )
    ht.wait()

    # centroid_t, cluster_sizes, per_sample_ineria change
    #### N.B.: Furtherst point from [.1, -.1, .1] is [-1, 1, -1] (Xnp[5]) so this should
    #### be new centroid chosen instead of [5,5,5]
    expected_updated_centroid_t = np.array([[0.11 + 1, -1], [-0.1 -1, 1], [0.09 + 1, -1]], dtype=dataT)
    expected_cluster_sizes = np.array([7, 1], dtype=dataT)
    expected_per_sample_inertia = np.copy(sq_dist_to_nearest_centroid_np)
    expected_per_sample_inertia[5] = 0

    assert_allclose(expected_cluster_sizes, dpt.asnumpy(cluster_sizes), rtol=np.finfo(dataT).resolution)
    assert_allclose(expected_updated_centroid_t, dpt.asnumpy(centroid_t), rtol=np.finfo(dataT).resolution)
    assert_allclose(expected_per_sample_inertia, dpt.asnumpy(per_sample_inertia), rtol=np.finfo(dataT).resolution)


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_relocate_empty_clusters_fp64():
    dataT = np.float64
    indT = np.int32
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

    Cnp = np.array([[0.11, -0.1, 0.09], [5, 5, 5]], dtype=dataT)
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
        work_group_size=256,
        sycl_queue = q
    )
    ht.wait()

    # centroid_t, cluster_sizes, per_sample_ineria change
    #### N.B.: Furtherst point from [.1, -.1, .1] is [-1, 1, -1] (Xnp[5]) so this should
    #### be new centroid chosen instead of [5,5,5]
    expected_updated_centroid_t = np.array([[0.11 + 1, -1], [-0.1 -1, 1], [0.09 + 1, -1]], dtype=dataT)
    expected_cluster_sizes = np.array([7, 1], dtype=dataT)
    expected_per_sample_inertia = np.copy(sq_dist_to_nearest_centroid_np)
    expected_per_sample_inertia[5] = 0

    assert_allclose(expected_cluster_sizes, dpt.asnumpy(cluster_sizes), rtol=np.finfo(dataT).resolution)
    assert_allclose(expected_updated_centroid_t, dpt.asnumpy(centroid_t), rtol=np.finfo(dataT).resolution)
    assert_allclose(expected_per_sample_inertia, dpt.asnumpy(per_sample_inertia), rtol=np.finfo(dataT).resolution)


def test_centroid_shifts_fp32():
    dataT = np.float32
    X1_t = dpt.asarray([[1,-5], [2,-4], [3, -3], [4,-2], [5, -1]], dtype=dataT)
    X2_t = dpt.asarray([[2,-4],[3,-3], [4,-2], [5,-1], [6,1]], dtype=dataT)

    cs = dpt.empty((X1_t.shape[1],), dtype=dataT)

    q = X1_t.sycl_queue

    ht, _ = kdp.compute_centroid_shifts_squared(
        X1_t, X2_t, cs, sycl_queue=q)
    ht.wait()

    assert np.allclose(
        dpt.asnumpy(cs),
        np.array([5, 8], dtype=dataT),
        rtol = np.finfo(dataT).resolution
    )


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_centroid_shifts_fp64():
    dataT = np.float64
    X1_t = dpt.asarray([[1,-5], [2,-4], [3, -3], [4,-2], [5, -1]], dtype=dataT)
    X2_t = dpt.asarray([[2,-4],[3,-3], [4,-2], [5,-1], [6,1]], dtype=dataT)

    cs = dpt.empty((X1_t.shape[1],), dtype=dataT)

    q = X1_t.sycl_queue

    ht, _ = kdp.compute_centroid_shifts_squared(
        X1_t, X2_t, cs, sycl_queue=q)
    ht.wait()

    assert np.allclose(
        dpt.asnumpy(cs),
        np.array([5, 8], dtype=dataT),
        rtol = np.finfo(dataT).resolution
    )


def test_compute_centroid_to_sample_distances_fp32():
    dataT = np.float32
    # compute_centroid_to_sample_distances(X_t, centroid_t, dm, work_group_size,
    # centroids_window_height, sycl_queue=q, depends=[]
    # )
    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    # create 8 random points around each p
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(8,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    dm = dpt.empty((centroid_t.shape[1], Xt.shape[1]), dtype=dataT)
    assert dm.flags.c_contiguous

    q = Xt.sycl_queue
    ht, _ = kdp.compute_centroid_to_sample_distances(
        Xt, centroid_t, dm, 256, 8, sycl_queue=q
    )
    ht.wait()

    dm_ref = np.sqrt(
        np.sum(np.square( Xnp_t[:, np.newaxis, :] - Cnt[:, :, np.newaxis] ), axis=0)
    )
    f = 64

    assert_allclose(
        dpt.asnumpy(dm),
        dm_ref,
        rtol = f*np.finfo(dataT).resolution,
    )


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_compute_centroid_to_sample_distances_fp64():
    dataT = np.float64
    # compute_centroid_to_sample_distances(X_t, centroid_t, dm, work_group_size,
    # centroids_window_height, sycl_queue=q, depends=[]
    # )
    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    # create 8 random points around each p
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(8,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    dm = dpt.empty((centroid_t.shape[1], Xt.shape[1]), dtype=dataT)
    assert dm.flags.c_contiguous

    q = Xt.sycl_queue
    ht, _ = kdp.compute_centroid_to_sample_distances(
        Xt, centroid_t, dm, 256, 8, sycl_queue=q
    )
    ht.wait()

    dm_ref = np.sqrt(
        np.sum(np.square( Xnp_t[:, np.newaxis, :] - Cnt[:, :, np.newaxis] ), axis=0)
    )
    f = 1

    assert_allclose(
        dpt.asnumpy(dm),
        dm_ref,
        rtol = f*np.finfo(dataT).resolution,
    )


def test_assignment_fp32():
    dataT = np.float32
    indT = np.int32
    cloud_size = 1734

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    q_alloc = dpctl.SyclQueue() #property="in_order")

    Xt = dpt.asarray(Xnp_t, dtype=dataT, sycl_queue=q_alloc)
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT, sycl_queue=q_alloc)
    assert centroid_t.flags.c_contiguous

    hl2n = dpt.empty(centroid_t.shape[1], dtype=dataT, sycl_queue=q_alloc)
    assigned_id = dpt.empty(Xt.shape[1], dtype=indT, sycl_queue=q_alloc)
    q = Xt.sycl_queue

    ht1, e_hl2n = kdp.half_l2_norm_squared(centroid_t, hl2n,
        work_group_size=256, sycl_queue=q)

    ht2, _ = kdp.assignment(
        Xt, centroid_t, hl2n, assigned_id,
        centroids_window_height = 8,
        work_group_size=256,
        sycl_queue=q,
        depends=[e_hl2n,]
    )

    ht1.wait()
    ht2.wait()

    expected_ids = np.repeat(np.arange(8, dtype=indT), cloud_size)

    assert_array_equal(expected_ids, dpt.asnumpy(assigned_id))


def test_assignment_fp32_duplicate():
    dataT = np.float32
    indT = np.int32
    cloud_size = 1734

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    q_alloc = dpctl.SyclQueue() #property="in_order")

    Xt = dpt.asarray(Xnp_t, dtype=dataT, sycl_queue=q_alloc)
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT, sycl_queue=q_alloc)
    assert centroid_t.flags.c_contiguous

    hl2n = dpt.empty(centroid_t.shape[1], dtype=dataT, sycl_queue=q_alloc)
    assigned_id = dpt.empty(Xt.shape[1], dtype=indT, sycl_queue=q_alloc)
    q = Xt.sycl_queue

    ht1, e_hl2n = kdp.half_l2_norm_squared(centroid_t, hl2n,
        work_group_size=256, sycl_queue=q)

    ht2, _ = kdp.assignment(
        Xt, centroid_t, hl2n, assigned_id,
        centroids_window_height = 8,
        work_group_size=256,
        sycl_queue=q,
        depends=[e_hl2n,]
    )

    ht1.wait()
    ht2.wait()

    expected_ids = np.repeat(np.arange(8, dtype=indT), cloud_size)

    assert_array_equal(expected_ids, dpt.asnumpy(assigned_id))


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_assignment_fp64():
    dataT = np.float64
    indT = np.int32
    cloud_size = 32

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    q_alloc = dpctl.SyclQueue() #property="in_order")
    Xt = dpt.asarray(Xnp_t, dtype=dataT, sycl_queue=q_alloc)
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT, sycl_queue=q_alloc)
    assert centroid_t.flags.c_contiguous

    hl2n = dpt.empty(centroid_t.shape[1], dtype=dataT, sycl_queue=q_alloc)
    assigned_id = dpt.empty(Xt.shape[1], dtype=indT, sycl_queue=q_alloc)
    q = Xt.sycl_queue

    ht1, e_hl2n = kdp.half_l2_norm_squared(centroid_t, hl2n,
        work_group_size=256, sycl_queue=q)

    ht2, _ = kdp.assignment(
        Xt, centroid_t, hl2n, assigned_id,
        centroids_window_height = 8,
        work_group_size=256,
        sycl_queue=q,
        depends=[e_hl2n,]
    )

    ht1.wait()
    ht2.wait()

    expected_ids = np.repeat(np.arange(8, dtype=indT), cloud_size)

    assert_array_equal(expected_ids, dpt.asnumpy(assigned_id))


def test_compute_inertia_fp32():
    dataT = np.float32
    indT = np.int32
    cloud_size = 32

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_samples = Xt.shape[1]
    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    _ids = np.repeat(np.arange(n_clusters, dtype=indT), cloud_size)
    assignment_id = dpt.asarray(_ids)
    sample_weight = dpt.ones(n_samples, dtype=dataT)
    per_sample_inertia = dpt.empty(n_samples, dtype=dataT)

    q = Xt.sycl_queue

    ht, _ = kdp.compute_inertia(
        Xt, sample_weight, centroid_t, assignment_id, per_sample_inertia,
        work_group_size=256,
        sycl_queue=q
    )
    ht.wait()

    assert Xnp_t.shape[1] == _ids.shape[0]
    expected_per_sample_inertia = \
        np.sum(np.square(Xnp_t - np.take_along_axis(Cnt, _ids[np.newaxis,:], axis=1)), axis=0)

    assert_allclose(
        expected_per_sample_inertia,
        dpt.asnumpy(per_sample_inertia),
        rtol=np.finfo(dataT).resolution
    )


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_compute_inertia_fp64():
    dataT = np.float64
    indT = np.int32
    cloud_size = 32

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_samples = Xt.shape[1]
    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    _ids = np.repeat(np.arange(n_clusters, dtype=indT), cloud_size)
    assignment_id = dpt.asarray(_ids)
    sample_weight = dpt.ones(n_samples, dtype=dataT)
    per_sample_inertia = dpt.empty(n_samples, dtype=dataT)

    q = Xt.sycl_queue

    ht, _ = kdp.compute_inertia(
        Xt, sample_weight, centroid_t, assignment_id, per_sample_inertia,
        work_group_size=256,
        sycl_queue=q
    )
    ht.wait()

    assert Xnp_t.shape[1] == _ids.shape[0]
    expected_per_sample_inertia = \
        np.sum(np.square(Xnp_t - np.take_along_axis(Cnt, _ids[np.newaxis,:], axis=1)), axis=0)

    assert_allclose(
        expected_per_sample_inertia,
        dpt.asnumpy(per_sample_inertia),
        rtol=np.finfo(dataT).resolution
    )


def test_reduce_vector_blocking_fp32():
    dataT = dpt.float32
    vec = dpt.arange(100, dtype=dataT)

    q = vec.sycl_queue
    res = kdp.reduce_vector_blocking(vec, sycl_queue=q)

    assert res == 50 * 99  # sum(k, 0 <= k < 100) == 100 * 99 / 2


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_reduce_vector_blocking_fp64():
    dataT = dpt.float64
    vec = dpt.arange(100, dtype=dataT)

    q = vec.sycl_queue
    res = kdp.reduce_vector_blocking(vec, sycl_queue=q)

    assert res == 50 * 99  # sum(k, 0 <= k < 100) == 100 * 99 / 2


def test_lloyd_single_step_fp32():
    dataT = dpt.float32
    indT = dpt.int32

    cloud_size = 16

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_features, n_samples = Xt.shape
    assert n_features == 3

    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    centroids_half_l2_norm = dpt.asarray(np.sum(np.square(Cnt), axis=0) / 2)

    _ids = np.repeat(np.arange(n_clusters, dtype=indT), cloud_size)
    assignment_id = dpt.empty(_ids.shape, dtype=_ids.dtype)
    sample_weight = dpt.ones(n_samples, dtype=dataT)

    n_copies = 4
    new_centroids_t_private_copies = dpt.zeros((n_copies, n_features, n_clusters,), dtype=dataT)
    cluster_sizes_private_copies = dpt.zeros((n_copies, n_clusters,), dtype=dataT)

    q = Xt.sycl_queue

    ht, _ = kdp.fused_lloyd_single_step(
        Xt, sample_weight, centroid_t, centroids_half_l2_norm, assignment_id,
        new_centroids_t_private_copies,
        cluster_sizes_private_copies,
        8,      # centroids_window_height
        256,    # work_group_size
        q       # sycl_queue
    )
    ht.wait()

    assert_array_equal(_ids, dpt.asnumpy(assignment_id))

    expected_cluster_sizes = np.sum(dpt.asnumpy(cluster_sizes_private_copies), axis=0)

    assert_allclose(
        expected_cluster_sizes,
        np.full((n_clusters, ), cloud_size, dtype=dataT),
        rtol = np.finfo(dataT).resolution
    )

    expected_new_centroid_t = np.reshape(Xnp_t, (n_features, n_clusters, cloud_size)).sum(axis=-1)
    actual_new_centroid_t = np.sum(dpt.asnumpy(new_centroids_t_private_copies), axis=0)

    assert_allclose(
        expected_new_centroid_t,
        actual_new_centroid_t,
        rtol = np.finfo(dataT).resolution
    )


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_lloyd_single_step_fp64():
    dataT = dpt.float64
    indT = dpt.int32

    cloud_size = 16

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    Xnp = np.concatenate([
        np.random.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_features, n_samples = Xt.shape
    assert n_features == 3

    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    centroid_t = dpt.asarray(Cnt, dtype=dataT)
    assert centroid_t.flags.c_contiguous

    centroids_half_l2_norm = dpt.asarray(np.sum(np.square(Cnt), axis=0) / 2)

    _ids = np.repeat(np.arange(n_clusters, dtype=indT), cloud_size)
    assignment_id = dpt.empty(_ids.shape, dtype=_ids.dtype)
    sample_weight = dpt.ones(n_samples, dtype=dataT)

    n_copies = 4
    new_centroids_t_private_copies = dpt.zeros((n_copies, n_features, n_clusters,), dtype=dataT)
    cluster_sizes_private_copies = dpt.zeros((n_copies, n_clusters,), dtype=dataT)

    q = Xt.sycl_queue

    ht, _ = kdp.fused_lloyd_single_step(
        Xt, sample_weight, centroid_t, centroids_half_l2_norm, assignment_id,
        new_centroids_t_private_copies,
        cluster_sizes_private_copies,
        8,      # centroids_window_height
        256,    # work_group_size
        q       # sycl_queue
    )
    ht.wait()

    assert_array_equal(_ids, dpt.asnumpy(assignment_id))

    expected_cluster_sizes = np.sum(dpt.asnumpy(cluster_sizes_private_copies), axis=0)

    assert_allclose(
        expected_cluster_sizes,
        np.full((n_clusters, ), cloud_size, dtype=dataT),
        rtol = np.finfo(dataT).resolution
    )

    expected_new_centroid_t = np.reshape(Xnp_t, (n_features, n_clusters, cloud_size)).sum(axis=-1)
    actual_new_centroid_t = np.sum(dpt.asnumpy(new_centroids_t_private_copies), axis=0)

    assert_allclose(
        expected_new_centroid_t,
        actual_new_centroid_t,
        rtol = np.finfo(dataT).resolution
    )


def test_kmeans_lloyd_driver_fp32():
    # kmeans_lloyd_driver(
    #    X_t, sample_weight, init_centroids_t, assignment_ids, res_centroids_t,
    #    tol, verbose, max_iters, centroids_window_height, work_group_size,
    #    centroids_private_copies_max_cache_occupancy,
    #    sycl_queue, depends=[]
    # )
    dataT = dpt.float32
    indT = dpt.int32

    cloud_size = 32

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    rs = np.random.default_rng(seed=12345)
    Xnp = np.concatenate([
        rs.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_features, n_samples = Xt.shape
    assert n_features == 3

    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    init_centroids_t = dpt.asarray(Cnt, dtype=dataT)
    assert init_centroids_t.flags.c_contiguous

    res_centroids_t = dpt.empty_like(init_centroids_t)
    sample_weight = dpt.ones(n_samples, dtype=dataT)
    assignment_ids = dpt.empty(n_samples, dtype=indT)

    q = Xt.sycl_queue

    tol = 1e-6
    max_iters = 255
    verbosity=False
    centroids_window_height = 8
    work_group_size = 128
    centroids_private_copies_max_cache_occupancy = 0.7

    n_iters_, total_inertia = kdp.kmeans_lloyd_driver(
        Xt, sample_weight, init_centroids_t, assignment_ids, res_centroids_t,
        tol, verbosity, max_iters, centroids_window_height, work_group_size,
        centroids_private_copies_max_cache_occupancy,
        q
    )

    expected_ids = np.repeat(np.arange(8, dtype=indT), cloud_size)
    assert_array_equal(expected_ids, dpt.asnumpy(assignment_ids))

    assert n_iters_ < max_iters
    assert n_iters_ == 2


@pytest.mark.skipif(_no_fp64_support(),
                    reason="Device does not support float64")
def test_kmeans_lloyd_driver_fp64():
    # kmeans_lloyd_driver(
    #    X_t, sample_weight, init_centroids_t, assignment_ids, res_centroids_t,
    #    tol, verbose, max_iters, centroids_window_height, work_group_size,
    #    centroids_private_copies_max_cache_occupancy,
    #    sycl_queue, depends=[]
    # )
    dataT = dpt.float64
    indT = dpt.int32

    cloud_size = 32

    ps = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=dataT)
    rs = np.random.default_rng(seed=12345)
    Xnp = np.concatenate([
        rs.normal(0, 0.1, size=(cloud_size,3)).astype(dataT) + p for p in ps
    ], axis=0)
    Xnp_t = np.ascontiguousarray(Xnp.T)
    Cnt = np.ascontiguousarray(ps.T)

    Xt = dpt.asarray(Xnp_t, dtype=dataT)
    n_features, n_samples = Xt.shape
    assert n_features == 3

    n_clusters = ps.shape[0]
    assert Xt.flags.c_contiguous
    init_centroids_t = dpt.asarray(Cnt, dtype=dataT)
    assert init_centroids_t.flags.c_contiguous

    res_centroids_t = dpt.empty_like(init_centroids_t)
    sample_weight = dpt.ones(n_samples, dtype=dataT)
    assignment_ids = dpt.empty(n_samples, dtype=indT)

    q = Xt.sycl_queue

    tol = 1e-6
    max_iters = 255
    verbosity=False
    centroids_window_height = 8
    work_group_size = 128
    centroids_private_copies_max_cache_occupancy = 0.7

    n_iters_, total_inertia = kdp.kmeans_lloyd_driver(
        Xt, sample_weight, init_centroids_t, assignment_ids, res_centroids_t,
        tol, verbosity, max_iters, centroids_window_height, work_group_size,
        centroids_private_copies_max_cache_occupancy,
        q
    )

    expected_ids = np.repeat(np.arange(8, dtype=indT), cloud_size)
    assert_array_equal(expected_ids, dpt.asnumpy(assignment_ids))

    assert n_iters_ < max_iters
    assert n_iters_ == 2
