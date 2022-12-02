#pragma once

#include <CL/sycl.hpp>
#include <vector>
#include <cstdint>
#include "quotients_utils.hpp"
#include "iterative_merge_sort.hpp"

template <typename T>
sycl::event 
broadcast_division_kernel(
    sycl::queue q,
    size_t n_features, 
    size_t n_clusters, 
    size_t, // work_group_size
    //
    T *new_centroids_t,        // IN & OUT  (n_features, n_clusters)
    T const *cluster_sizes,    // IN        (n_clusters,)
    const std::vector<sycl::event> &depends = {}
) {

    // FIXME: let kernel process several elements of centroid matrix
    sycl::event res_ev = q.submit(
        [&] (sycl::handler &cgh) {
            cgh.depends_on(depends);
            auto gwsRange = sycl::range<1>(n_features * n_clusters);
            cgh.parallel_for(
                gwsRange,
                [=](sycl::id<1> wid) {
                    auto i = wid[0];
                    size_t cluster_id = i / n_features;
                    size_t feature_id = i - cluster_id * n_features;

                    auto offset = feature_id * n_clusters + cluster_id;
                    new_centroids_t[offset] /= cluster_sizes[cluster_id];
                }
            );
        });

    return res_ev;
}

template <typename T>
class half_l2_norm_krn;

// centroids_half_l2_norm_squared = np.square(centroids_t).sum(axis=0) / 2 
template <typename T>
sycl::event
half_l2_norm_kernel(
    sycl::queue q,
    size_t n_features,    // size0
    size_t n_clusters,    // size1
    size_t work_group_size,
    //
    T const *centroids_t,              // IN  (n_features, n_clusters)
    T *centroids_half_l2_norm_squared, // OUT (n_clusters)
    const std::vector<sycl::event> &depends = {}
) {
    // FIXME: write it more efficiently
    sycl::event res_ev = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            size_t global_size = quotient_ceil(n_clusters, work_group_size) * work_group_size;
            cgh.parallel_for<class half_l2_norm_krn<T>>(
                sycl::nd_range<1>(global_size, work_group_size),
                [=](sycl::nd_item<1> it) {
                    auto col_idx = it.get_global_linear_id();
                    if (col_idx < n_clusters) {
                        T l2_norm(0);
                        for(size_t row_idx=0; row_idx < n_features; ++row_idx) {
                            T item = centroids_t[n_clusters * row_idx + col_idx];
                            l2_norm += item * item;
                        }

                        centroids_half_l2_norm_squared[col_idx] = l2_norm / T(2);
                    }
                }
            );
        });

    return res_ev;
}

template<typename dataT, typename indT>
class reduce_centroid_data_krn;

template<typename dataT, typename indT>
sycl::event 
reduce_centroid_data_kernel(
    sycl::queue q,
    size_t n_centroids_private_copies,
    size_t n_features,
    size_t n_clusters,
    size_t work_group_size,
    //
    dataT const *cluster_sizes_private_copies, // IN  (n_copies, n_clusters)
    dataT const *centroids_t_private_copies,   // IN  (n_copies, n_features, n_clusters)
    dataT *cluster_sizes,         // OUT  (n_clusters)
    dataT *centroids_t,           // OUT  (n_features, n_clusters,)
    indT *empty_clusters_list,    // OUT  (n_clusters,)
    indT *n_empty_clusters,       // OUT  (1,)
    const std::vector<sycl::event> &depends = {}
) {

    sycl::event res_ev = 
        q.submit([&] (sycl::handler &cgh) {
            cgh.depends_on(depends);

            size_t n_work_groups_for_clusters = 
                quotient_ceil(n_clusters, work_group_size);
            size_t n_work_items_for_clusters = n_work_groups_for_clusters * work_group_size;
            size_t gws = n_work_items_for_clusters * n_features;

            cgh.parallel_for<class reduce_centroid_data_krn<dataT, indT>>(
                sycl::nd_range<1>({gws}, {work_group_size}),
                [=](sycl::nd_item<1> it) {
                    size_t group_idx = it.get_group(0);
                    size_t item_idx = it.get_local_linear_id();
                    size_t feature_idx = group_idx / n_work_groups_for_clusters;
                    size_t cluster_idx = item_idx + (
                        (group_idx % n_work_groups_for_clusters) * work_group_size 
                    );

                    if (cluster_idx < n_clusters) {
                        {
                            dataT sum_(0);
                            size_t offset = feature_idx * n_clusters + 
                                    cluster_idx;
                            size_t step = n_features * n_clusters;
                            for(size_t copy_idx = 0; copy_idx < n_centroids_private_copies; ++copy_idx) {
                                sum_ += centroids_t_private_copies[copy_idx * step + offset];
                            }
                            centroids_t[offset] = sum_;
                        }

                        if (feature_idx == 0) {
                            dataT sum_(0);
                            for(size_t copy_idx = 0; copy_idx < n_centroids_private_copies; ++copy_idx) {
                                sum_ += cluster_sizes_private_copies[copy_idx * n_clusters + cluster_idx];
                            }
                            cluster_sizes[cluster_idx] = sum_;

                            // FIXME: this is race condition
                            if (sum_ == 0) {
                                sycl::atomic_ref<
                                    indT,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> v(
                                        n_empty_clusters[0]);
                                indT i = v.fetch_add(indT(1));
                                empty_clusters_list[i] = cluster_idx;
                            }
                        }
                    }
                }
            );
        });

    return res_ev;
}

template <typename dataT>
sycl::event
compute_threshold_kernel(
    sycl::queue q,
    size_t n_samples,
    dataT const *data,
    size_t n_empty_clusters,
    dataT *threshold,
    const std::vector<sycl::event> &depends={}
)
{
    if (n_empty_clusters == 1) {
        sycl::event res_ev = 
            q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                sycl::property_list prop( {sycl::property::reduction::initialize_to_identity{}} );
                auto maxReduction = sycl::reduction(threshold, sycl::maximum<dataT>(), prop);
                cgh.parallel_for(
                    sycl::range<1>(n_samples), 
                    maxReduction, 
                    [=] (sycl::id<1> wid, auto &max) {
                        max.combine(data[wid]);
                    }
                );
            });
        return res_ev;
    } else {
        dataT *temp_output = sycl::malloc_device<dataT>(n_samples, q);

        sycl::event sort_ev = 
            iterative_merge_sort(q, data, temp_output, n_samples, depends);

        sycl::event copy_ev = q.copy<dataT>(temp_output + n_samples - n_empty_clusters, threshold, 1, {sort_ev});

        // asynchronously free temporary
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(copy_ev);
            auto ctx = q.get_context();
            cgh.host_task([ctx, temp_output] { sycl::free(temp_output, ctx); });
        });

        return copy_ev;
    }
}

template <typename dataT, typename indT>
class select_samples_far_from_cetrnoids_krn;

template <typename dataT, typename indT>
sycl::event
select_samples_far_from_centroid_kernel(
    sycl::queue q,
    size_t n_empty_clusters, 
    size_t n_samples, 
    size_t work_group_size,
    //
    dataT const *distance_to_centroid, // IN (n_samples,)
    dataT const *threshold,            // IN (1, )
    indT *selected_samples_idx,        // OUT (n_samples,)
    indT *n_selected_gt_threshold,     // OUT (1,)
    indT *n_selected_eq_threshold,     // OUT (1,)
    const std::vector<sycl::event> &depends = {}
) {
    /* 
    This kernel writes in selected_samples_idx the idx of the top n_selected
    items in distance_to_centroid with highest values.

    threshold is expected to have been pre-computed (by partitioning) such that
    there are at most `n_selected-1` values that are strictly greater than
    threshold, and at least n_selected values that are greater or equal than
    threshold.

    Because the exact number of values strictly equal to the threshold is not known
    and that the goal is to select the top n_selected greater items above threshold,
    we write indices of values strictly greater than threshold at the beginning of
    the selected_samples_idx, and indices of values equal to threshold at the end
    of the selected_samples_idx array.
    */

    sycl::event res_ev = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.parallel_for<class select_samples_far_from_cetrnoids_krn<dataT, indT>>(
                {n_samples},
                [=](sycl::id<1> wid) {
                    size_t sample_idx = wid[0];
                    if (sample_idx >= n_samples) 
                        return;

                    indT n_selected_gt_threshold_ = n_selected_gt_threshold[0];
                    indT n_selected_eq_threshold_ = n_selected_eq_threshold[0];
                    indT max_n_selected_gt_threshold = n_empty_clusters - 1;
                    indT min_n_selected_eq_threshold = 2;
                    indT max_n_selected_eq_threshold = n_empty_clusters + 1;

                    if ((n_selected_gt_threshold_ == max_n_selected_gt_threshold) && (n_selected_eq_threshold_ >= min_n_selected_eq_threshold))
                        return;

                    dataT threshold_ = threshold[0];
                    dataT distance_to_centroid_ = distance_to_centroid[sample_idx];

                    if (distance_to_centroid_ < threshold_) {
                        return;
                    } else if(distance_to_centroid_ > threshold_) {
                        auto atomic_n_selected_gt_threshold = 
                        sycl::atomic_ref<
                            indT, 
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(n_selected_gt_threshold[0]);
                        indT selected_idx = atomic_n_selected_gt_threshold.fetch_add(1);
                        selected_samples_idx[selected_idx] = sample_idx;

                        return;
                    } else {
                        if (n_selected_eq_threshold_ >= max_n_selected_eq_threshold) 
                            return;

                        auto atomic_n_selected_eq_threshold = 
                        sycl::atomic_ref<
                            indT, 
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(n_selected_eq_threshold[0]);

                        indT selected_idx = n_samples - atomic_n_selected_eq_threshold.fetch_add(1);
                        selected_samples_idx[selected_idx] = sample_idx;
                    }
                }
            );
        });

    return res_ev;
}

template <typename dataT, typename indT>
class relocate_empty_clusters_krn;

template <typename dataT, typename indT>
sycl::event 
relocate_empty_clusters_kernel(
    sycl::queue q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    size_t n_empty_clusters,
    indT *n_selected_gt_threshold,   // USM pointer
    size_t work_group_size,
    //
    dataT const *X_t,                  // IN, READ ONLY (n_features, n_samples,)
    dataT const *sample_weight,        // IN, READ ONLY (n_samples,)
    indT const *assignment_id,            // IN  (n_samples,)
    indT const *samples_far_from_center,  // IN  (n_samples, )
    indT const *empty_clusters_list,   // IN  (n_clusters, )
    dataT *per_sample_inertia,         // INOUT (n_samples,)
    dataT *centroids_t,                // INOUT (n_features, n_clusters,)  
    dataT *cluster_sizes,              // INOUT (n_clusters,)
    const std::vector<sycl::event> &depends = {}
) 
{
    sycl::event res_ev = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            size_t n_work_groups_for_cluster = quotient_ceil(n_features, work_group_size);
            size_t n_work_items_for_cluster = n_work_groups_for_cluster * work_group_size;
            size_t global_size = n_work_items_for_cluster * n_relocated_clusters;

            cgh.parallel_for<class relocate_empty_clusters_krn<dataT, indT>>(
                sycl::nd_range<1>({global_size}, {work_group_size}),
                [=](sycl::nd_item<1> wit) {
                    size_t group_idx = wit.get_group(0);
                    size_t item_idx = wit.get_local_id(0);

                    size_t relocated_idx = group_idx / n_work_groups_for_cluster;
                    size_t feature_idx = (group_idx - relocated_idx * n_work_groups_for_cluster) * work_group_size + item_idx;

                    if (feature_idx >= n_features) return;

                    indT n_selected_gt_threshold_ = n_selected_gt_threshold[0] - 1;
                    indT relocated_cluster_idx = empty_clusters_list[relocated_idx];
                    indT new_location_X_idx = samples_far_from_center[n_selected_gt_threshold_ - relocated_idx];
                    indT new_location_previous_assignment = assignment_id[new_location_X_idx];

                    dataT new_centroid_value = X_t[feature_idx * n_samples + new_location_X_idx];
                    dataT new_location_weight = sample_weight[new_location_X_idx];
                    dataT X_centroid_addend = new_centroid_value * new_location_weight;

                    auto atomic_centroid_component_ref = 
                    sycl::atomic_ref<
                            dataT, 
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(centroids_t[feature_idx * n_clusters + new_location_previous_assignment]);

                    atomic_centroid_component_ref -= X_centroid_addend;
                    centroids_t[feature_idx * n_clusters + relocated_cluster_idx] = X_centroid_addend;

                    if (feature_idx == 0) {
                        per_sample_inertia[new_location_X_idx] = dataT(0);
                        auto atomic_cluster_size_ref = 
                        sycl::atomic_ref<
                                dataT, 
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(cluster_sizes[new_location_previous_assignment]);
                        atomic_cluster_size_ref -= new_location_weight;
                        cluster_sizes[relocated_cluster_idx] = new_location_weight;
                    }
                }
            );
        });

    return res_ev;
}

template <typename dataT, typename indT>
sycl::event 
relocate_empty_clusters(
    sycl::queue q,
    size_t n_samples, 
    size_t n_features, 
    size_t n_clusters,
    size_t work_group_size,
    //
    size_t n_empty_clusters,
    dataT const *X_t,                          // IN (n_features, n_samples)
    dataT const *sample_weight,                // IN (n_samples, )
    indT const *assignment_id,                 // IN (n_samples, )
    indT const *empty_clusters_list,           // IN (n_clusters, )
    dataT const *sq_dist_to_nearest_centroid,  // IN (n_samples, )
    dataT *centroids_t,                        // INOUT (n_features, n_clusters)
    dataT *cluster_sizes,                      // INOUT (n_clusters,)
    dataT *per_sample_inertia,                 // INOUT (n_sample, )
    const std::vector<sycl::event> &depends = {}
) {
    size_t kth = n_samples - n_empty_clusters;

    // compute threshold = kth largest element in sq_dist_to_nearest_centroid
    dataT *threshold = sycl::malloc_device<dataT>(1, q);

    sycl::event compute_threshold_ev = 
        compute_threshold_kernel(q, n_samples, sq_dist_to_nearest_centroid, n_empty_clusters, threshold, depends);

    indT *samples_far_from_center = sycl::malloc_device<indT>(n_samples + 2, q);
    indT *n_selected = samples_far_from_center + n_samples;

    indT *n_selected_gt_threshold = n_selected;
    indT *n_selected_eq_threshold = n_selected + 1;

    indT zero_one[2] = {0, 1};
    sycl::event n_selected_pop_ev = q.copy<indT>(&zero_one[0], n_selected, 2);

    sycl::event select_samples_far_from_centroid_ev = 
        select_samples_far_from_centroid_kernel<dataT, indT>(
            q,
            n_empty_clusters, n_samples, work_group_size,
            //
            sq_dist_to_nearest_centroid, // IN (n_samples,)
            threshold,                   // IN (1, )
            samples_far_from_center,     // OUT (n_samples,)
            n_selected_gt_threshold,     // OUT (1,)
            n_selected_eq_threshold,     // OUT (1,)
            {compute_threshold_ev, n_selected_pop_ev}
        );

    sycl::event relocate_empty_cluster_ev = 
        relocate_empty_clusters_kernel<dataT, indT>(
            q,
            n_samples,
            n_features,
            n_clusters,
            n_empty_clusters,
            n_selected_gt_threshold,   // USM pointer
            work_group_size,
            //
            X_t,                                 // IN, READ ONLY (n_features, n_samples,)
            sample_weight,                       // IN, READ ONLY (n_samples,)
            assignment_id,                       // IN  (n_samples,)
            samples_far_from_center,             // IN  (n_samples, )
            empty_clusters_list,                 // IN  (n_clusters, )
            per_sample_inertia,                  // INOUT (n_samples,)
            centroids_t,                         // INOUT (n_features, n_clusters,)  
            cluster_sizes,                       // INOUT (n_clusters,)
            {select_samples_far_from_centroid_ev}  
        );

    return relocate_empty_cluster_ev;
}

template <typename dataT>
class compute_centroid_shifts_krn;

template <typename dataT>
sycl::event
compute_centroid_shifts_squared_kernel(
    sycl::queue q,
    size_t n_features, 
    size_t n_clusters, 
    size_t work_group_size,
    //
    dataT const *centroids_t,     // IN
    dataT const *new_centroids_t, // IN
    dataT *centroid_shifts,       // OUT 
    const std::vector<sycl::event> &depends = {}
) {
    sycl::event res_ev = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            // FIXME: write to have more parallelism
            cgh.parallel_for<class compute_centroid_shifts_krn<dataT>>(
                {n_clusters},
                [=](sycl::id<1> wid) {
                    size_t cluster_idx = wid[0];

                    dataT squared_centroid_diff(0);
                    for(size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
                        auto linear_id = feature_idx * n_clusters + cluster_idx;
                        dataT center_diff = centroids_t[linear_id ] - new_centroids_t[linear_id];
                        squared_centroid_diff += center_diff * center_diff;
                    }
                    centroid_shifts[cluster_idx] = squared_centroid_diff;
                }
            );
        });

    return res_ev;
}
