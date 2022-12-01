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
