// lloyd_single_step.hpp

#pragma once
#include <CL/sycl.hpp>
#include <vector>

#include "quotients_utils.hpp"

template <typename T, typename indT, size_t preferred_work_group_size_multiple, size_t centroids_window_width_multiplier>
class lloyd_single_step_krn;

template <typename T, size_t preferred_work_group_size_multiple, size_t centroids_window_width_multiplier>
size_t compute_number_of_private_copies(
    sycl::queue q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    double centroids_private_copies_max_cache_occupancy,
    size_t work_group_size
) {
    size_t global_mem_cache_size = q.get_device().get_info<sycl::info::device::global_mem_cache_size>();

    size_t n_cluster_items = n_clusters * (n_features + 1);
    size_t n_cluster_bytes = sizeof(T) * n_cluster_items;

    size_t n_centroids_private_copies = static_cast<size_t>(
        (global_mem_cache_size * centroids_private_copies_max_cache_occupancy) / n_cluster_bytes);

    size_t global_size = quotient_ceil(n_samples, work_group_size) * work_group_size;
    size_t n_subgroups = global_size / preferred_work_group_size_multiple;

    n_centroids_private_copies = std::min(n_subgroups, n_centroids_private_copies);

    return n_centroids_private_copies;
}

template <typename T, typename indT, size_t preferred_work_group_size_multiple, size_t centroids_window_width_multiplier>
sycl::event
lloyd_single_step(
    sycl::queue q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    size_t centroids_window_height,
    size_t n_centroids_private_copies,
    size_t work_group_size,
    // ===================
    const T *X_t,                      // IN READ-ONLY  (n_features, n_samples)
    const T *sample_weights,           // IN READ_ONLY  (n_samples, )   ????
    const T *current_centroids_t,      // IN            (n_features, n_clusters)
    const T *centroids_half_l2_norm,   // IN            (n_clusters, )
    indT *assignments_idx,             // OUT           (n_samples, )
    T *new_centroids_t_private_copies, // OUT           (n_private_copies, n_features, n_clusters)
    T *cluster_sizes_private_copies,   // OUT           (n_private_copies, n_clusters)  # noqa
    const std::vector<sycl::event> &depends = {}
)
{
    constexpr size_t window_n_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    );
    constexpr T inf = std::numeric_limits<T>::infinity();

    size_t n_windows_for_centroid = quotient_ceil(n_clusters, window_n_centroids);
    size_t n_windows_for_feature = quotient_ceil(n_features, centroids_window_height);

    size_t global_size = quotient_ceil(n_samples, work_group_size) * work_group_size;

    sycl::event e =
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            auto G = sycl::range<1>(global_size);
            auto L = sycl::range<1>(work_group_size);

            sycl::stream out(1024, 256, cgh);

            // allocate SLM
            using slm_cwT = sycl::local_accessor<T, 2>;
            slm_cwT centroids_window(sycl::range<2>(centroids_window_height, (window_n_centroids + 1)), cgh);

            using slm_l2hnT = sycl::local_accessor<T, 1>;
            slm_l2hnT window_of_centroids_half_l2_norms(sycl::range<1>(window_n_centroids), cgh);

            cgh.parallel_for<class lloyd_single_step_krn<T, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>>(
                sycl::nd_range<1>(G, L),
                [=](sycl::nd_item<1> it) {
                    size_t sample_idx = it.get_global_id(0);
                    size_t local_work_id = it.get_local_id(0);

                    std::array<T, window_n_centroids> dot_products;

                    size_t first_centroid_idx = 0;
                    size_t min_idx = 0;
                    T min_sample_pseudo_inertia(inf);

                    size_t window_loading_feature_offset = local_work_id / window_n_centroids;
                    size_t window_loading_centroid_idx = local_work_id - window_n_centroids * window_loading_feature_offset;

                    for(size_t i0 = 0; i0 < n_windows_for_centroid; ++i0) {
                         _initialize_window_of_centroids<T>(
                            n_clusters,
                            n_features,
                            work_group_size,
                            window_n_centroids,
                            centroids_window_height,
                            // ======================
                            local_work_id,
                            first_centroid_idx,
                            centroids_half_l2_norm,
                            window_of_centroids_half_l2_norms,
                            dot_products
                        );

                        size_t loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx;

                        size_t first_feature_idx = 0;

                        for(size_t i1 = 0; i1 < n_windows_for_centroid; ++i1) {
                            _load_window_of_centroids_and_features(
                                n_clusters,
                                n_features,
                                work_group_size,
                                window_n_centroids,
                                centroids_window_height,
                                // =====
                                first_feature_idx,
                                loading_centroid_idx,
                                window_loading_centroid_idx,
                                window_loading_feature_offset,
                                current_centroids_t,
                                centroids_window
                            );

                            it.barrier(sycl::access::fence_space::local_space);

                            constexpr bool acummulate_dot_product = true;
                            _acummulate_sum_of_ops<T, decltype(centroids_window), decltype(dot_products), acummulate_dot_product>(
                                n_samples,
                                n_features,
                                centroids_window_height,
                                window_n_centroids,
                                // ==============
                                sample_idx,
                                first_feature_idx,
                                X_t,
                                centroids_window,
                                dot_products
                            );

                            it.barrier(sycl::access::fence_space::local_space);

                            first_feature_idx += centroids_window_height;
                        }

                        auto closest = _update_closest_centroid(
                            window_n_centroids,
                            // =================
                            first_centroid_idx,
                            min_idx,
                            min_sample_pseudo_inertia,
                            window_of_centroids_half_l2_norms,
                            dot_products.data()
                        );

                        it.barrier(sycl::access::fence_space::local_space);

                        min_idx = closest.first;
                        min_sample_pseudo_inertia = closest.second;
                    }

                    if (sample_idx < n_samples) {
                        assignments_idx[sample_idx] = min_idx;

                        T weight = sample_weights[sample_idx];

                        size_t privatization_idx = (
                            sample_idx / preferred_work_group_size_multiple
                        ) % n_centroids_private_copies;

                        auto atomic_cluser_size =
                        sycl::atomic_ref<
                            T,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                cluster_sizes_private_copies[privatization_idx * n_clusters + min_idx]
                            );

                        atomic_cluser_size += weight;

                        // new_centroids_t_private_copies  (n_copies, n_features, n_clusters)
                        size_t _offset = privatization_idx * n_features * n_clusters + min_idx;
                        for(size_t feature_idx = 0; feature_idx < n_features; ++feature_idx ) {
                            auto atomic_coord =
                            sycl::atomic_ref<
                                T,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                    new_centroids_t_private_copies[_offset + feature_idx * n_clusters]
                                );

                            atomic_coord += X_t[feature_idx * n_samples + sample_idx ] * weight;
                        }
                    }
                }
            );
        });

    return e;
}
