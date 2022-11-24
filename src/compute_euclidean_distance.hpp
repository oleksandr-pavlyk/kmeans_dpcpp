// kernels for computing euclidean distance

#pragma once
#include <CL/sycl.hpp>
#include "device_functions.hpp"
#include <vector>
#include "quotients_utils.hpp"

template <typename T, size_t preferred_work_group_size_multiplier, size_t centroids_window_width_multiplier> 
class euclidean_distance_krn;

template <typename T, size_t preferred_work_group_size_multiplier, size_t centroids_window_width_multiplier>
sycl::event
compute_distances(
    sycl::queue q,
    // ==================
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    size_t centroids_window_height,
    size_t work_group_size,
    // ====================
    const T *X_t,                   // (n_features, n_samples)
    const T *current_centroids_t,   // (n_features, n_clusters)
    T *euclidean_distances_t,       // (n_clusters, n_samples)
    const std::vector<sycl::event> &depends = {}
) {
    constexpr size_t window_n_centroids = 
        preferred_work_group_size_multiplier * centroids_window_width_multiplier;

    size_t n_windows_for_centroid = quotient_ceil(n_clusters, window_n_centroids);
    size_t n_windows_for_feature = quotient_ceil(n_features, centroids_window_height);

    sycl::event e = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            auto G = sycl::range<1>( quotient_ceil(n_samples, work_group_size) * work_group_size);
            auto L = sycl::range<1>( work_group_size );

            // allocate SLM
            using slmT = sycl::accessor<T, 2, sycl::access::mode::read_write, sycl::access::target::local>;
            slmT centroids_window(sycl::range<2>(centroids_window_height, (window_n_centroids + 1)), cgh);

            cgh.parallel_for<class euclidean_distance_krn<T, preferred_work_group_size_multiplier, centroids_window_width_multiplier>>(
                sycl::nd_range<1>(G, L),
                [=](sycl::nd_item<1> it) {
                    size_t sample_idx = it.get_global_id(0);
                    size_t local_work_id = it.get_local_id(0);

                    std::array<T, window_n_centroids> sq_distances;
                    size_t window_loading_feature_offset = local_work_id / window_n_centroids;
                    size_t window_loading_centroid_idx = local_work_id - window_n_centroids * window_loading_feature_offset;

                    size_t first_centroid_idx = 0;

                    for(size_t i0 = 0; i0 < n_windows_for_feature; ++i0) {
                        _initialize_results<T>(
                            n_clusters, n_features, work_group_size, window_n_centroids, centroids_window_height, 
                            sq_distances);

                        size_t loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx;
                        size_t first_feature_idx = 0;

                        for(size_t i1 = 0; i1 < n_windows_for_centroid; ++i1) {
                            _load_window_of_centroids_and_features<T>(
                                n_clusters,
                                n_features,
                                work_group_size,
                                window_n_centroids,
                                centroids_window_height,
                                // =============
                                first_feature_idx,
                                loading_centroid_idx,
                                window_loading_centroid_idx,
                                window_loading_feature_offset,
                                current_centroids_t,
                                centroids_window
                            );

                            it.barrier(sycl::access::fence_space::local_space);

                            constexpr bool accumulate_distance_squared = false;
                            _acummulate_sum_of_ops<T, slmT, decltype(sq_distances), accumulate_distance_squared>(
                                n_samples, n_features, centroids_window_height, window_n_centroids,
                                // ======================
                                sample_idx, first_feature_idx, X_t, centroids_window, sq_distances
                            );

                            it.barrier(sycl::access::fence_space::local_space);

                            first_feature_idx += centroids_window_height;

                        }

                        if (sample_idx < n_samples) {
                            for(size_t i = 0; i < window_n_centroids; ++i) {
                                size_t centroid_idx = first_centroid_idx + i;
                                if (centroid_idx < n_clusters) {
                                     euclidean_distances_t[centroid_idx * n_samples + sample_idx] = 
                                        sycl::sqrt(sq_distances[i]);
                                }
                            }
                        }

                        it.barrier(sycl::access::fence_space::local_space);

                        first_centroid_idx += window_n_centroids;
                    }
                }
            );
        });

    return e;
}