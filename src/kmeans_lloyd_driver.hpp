#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <sstream>

#include "quotients_utils.hpp"
#include "device_functions.hpp"
#include "lloyd_single_step.hpp"
#include "compute_inertia.hpp"
#include "assignment.hpp"
#include "compute_euclidean_distance.hpp"
#include "util_kernels.hpp"

/* @brief Computes lloyd iterations
   Returns n_iteration
 */
template <typename dataT, typename indT = std::uint32_t, size_t preferred_work_group_size_multiple, size_t centroids_window_width_multiplier, typename PrintFuncT>
size_t driver_lloyd(
    sycl::queue exec_q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    // all things from self
    double centroids_private_copies_max_cache_occupancy,
    size_t centroids_window_height,
    size_t work_group_size,
    // inputs
    dataT const *X_t,
    dataT const *sample_weight,
    dataT *init_centroids_t,
    size_t max_iter,
    bool verbose,
    dataT tol,
    // outputs
    indT *assignment_id,
    dataT *res_centroids_t,
    dataT &total_inertia,
    PrintFuncT print_func
)
{
    const auto &alloc_ctx = exec_q.get_context();
    const auto &alloc_dev = exec_q.get_device();
  
    // USM temporary allocations, scheduled to be freed when computations complete
    dataT *centroids_half_l2_norm = sycl::malloc_device<dataT>(n_clusters, alloc_dev, alloc_ctx);

    dataT *cluster_sizes = sycl::malloc_device<dataT>(n_clusters, alloc_dev, alloc_ctx);
    dataT *centroid_shifts = sycl::malloc_device<dataT>(n_clusters, alloc_dev, alloc_ctx);

    // NB: the same buffer is used for those two arrays because it is never needed
    // to store those simultaneously in memory.
    dataT *per_sample_inertia = sycl::malloc_device<dataT>(n_samples, alloc_dev, alloc_ctx);
    dataT *sq_distance_to_nearest_centroid = per_sample_inertia;

    size_t n_centroids_private_copies = 
        compute_number_of_private_copies<dataT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
            exec_q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, work_group_size
        );

    size_t new_centroids_t_private_copies_size =
        n_centroids_private_copies * n_features * n_clusters; 
    dataT *new_centroids_t_private_copies = sycl::malloc_device<dataT>( 
        new_centroids_t_private_copies_size, alloc_dev, alloc_ctx);

    size_t cluster_sizes_private_copies_size = 
        n_centroids_private_copies * n_clusters;
    dataT *cluster_sizes_private_copies = sycl::malloc_device<dataT>(
        cluster_sizes_private_copies_size, alloc_dev, alloc_ctx);

    indT *empty_clusters_list = sycl::malloc_device<indT>(n_clusters + 1, alloc_dev, alloc_ctx);
    indT *n_empty_clusters = empty_clusters_list + n_clusters;

    size_t n_iterations = 0;
    dataT centroid_shifts_sum = std::numeric_limits<dataT>::infinity();

    dataT *this_centroids_t = init_centroids_t;
    dataT *new_centroids_t = res_centroids_t;

    while( (n_iterations < max_iter) && (centroid_shifts_sum > tol) ) {

        // populate centroids_half_norm
        sycl::event half_l2_norm_ev = half_l2_norm_kernel<dataT>(
            exec_q,
            n_features, n_clusters, work_group_size,
            //
            this_centroids_t, 
            centroids_half_l2_norm);

        // zero out cluster_sizes_private_copies
        sycl::event reset_cluster_sizes_private_copies_ev =
            exec_q.fill<dataT>(
                cluster_sizes_private_copies, 
                dataT(0), 
                cluster_sizes_private_copies_size
            );

        // zero out new_centroids_t_private_copies
        sycl::event reset_centroids_private_copies_ev = 
            exec_q.fill<dataT>(
                new_centroids_t_private_copies,
                dataT(0),
                new_centroids_t_private_copies_size
            );

        // n_empty_clusters[0] = np.int32(0)
        sycl::event set_n_empty_clusters_ev = 
            exec_q.fill<indT>(n_empty_clusters, indT(0), 1);

        /*
            fused_lloyd_fixed_window_single_step_kernel(
                X_t,
                sample_weight,
                centroids_t,
                centroids_half_l2_norm,
                assignments_idx,
                new_centroids_t_private_copies,
                cluster_sizes_private_copies,
            )
        */
        sycl::event lloyd_step_ev = 
            lloyd_single_step<
                dataT, indT, preferred_work_group_size_multiple, 
                centroids_window_width_multiplier
            >(
                exec_q, 
                n_samples, n_features, n_clusters,
                centroids_window_height,
                n_centroids_private_copies,
                work_group_size,
                // 
                X_t, 
                sample_weight,
                this_centroids_t,
                centroids_half_l2_norm,
                assignment_id,                    // OUT
                new_centroids_t_private_copies,   // OUT
                cluster_sizes_private_copies,     // OUT
                {half_l2_norm_ev, reset_centroids_private_copies_ev, reset_cluster_sizes_private_copies_ev}
            );

        /* 
        reduce_centroid_data_kernel(
                cluster_sizes_private_copies,
                new_centroids_t_private_copies,
                cluster_sizes,
                new_centroids_t,
                empty_clusters_list,
                n_empty_clusters,
            )
        */
        sycl::event reduce_centroid_data_ev = 
            reduce_centroid_data_kernel<dataT, indT>(
                exec_q, 
                n_centroids_private_copies,
                n_features, 
                n_clusters,
                work_group_size,
                //
                cluster_sizes_private_copies,    // OUT  (n_copies, n_clusters)
                new_centroids_t_private_copies,  // OUT  (n_copies, n_features, n_clusters)
                cluster_sizes,         // OUT  (n_clusters)
                new_centroids_t,       // OUT  (n_features, n_clusters,)
                empty_clusters_list,   // OUT  (n_clusters,)
                n_empty_clusters,      // OUT  (1,)
                {lloyd_step_ev}
            );

        if (verbose) {
            // auto compute_inertia_ev = compute_inertia_kernel<dataT>(exec_q, 
            // X_t, sample_weight, new_centroids_t, assignment_idx, per_sample_inertia,
            // {reduce_centroid_data_ev});

            // auto interia_reduce_ev = reduce_inertia_kernel<dataT>(
            //     exec_q, per_sample_inertia, {compute_inertial_ev});
            sycl::event compute_inertia_ev = 
                compute_inertia_kernel<dataT>(
                    exec_q,
                    n_samples, n_features, n_clusters, work_group_size,
                    //
                    X_t, sample_weight,
                    new_centroids_t,
                    assignment_id,
                    per_sample_inertia,
                    {reduce_centroid_data_ev}
                );

            dataT iteration_total_inertia =
                reduce_vector_kernel_blocking<dataT>(
                    exec_q,
                    n_samples,
                    per_sample_inertia,
                    {compute_inertia_ev} 
                );

            std::stringstream ss;
            ss << "Iteration: " << n_iterations << " "
               << "Inertia: " << iteration_total_inertia 
               << std::endl;

            print_func(ss);
        }

        indT host_n_empty_clusters;

        sycl::event n_empty_clusters_copy_ev = 
            exec_q.copy<indT>(n_empty_clusters, &host_n_empty_clusters, 1, {reduce_centroid_data_ev});
        n_empty_clusters_copy_ev.wait();

        // n_empty_clusters_ = int(n_empty_clusters[0])

        sycl::event relocate_empty_clusters_ev{};

        if (host_n_empty_clusters > 0) {
            /* 
              NB: empty cluster very rarely occurs, and it's more efficient to
              compute inertia and labels only after occurrences have been detected
              at the cost of an additional pass on data, rather than computing
              inertia by default during the first pass on data in case there's an
              empty cluster.
            */

            sycl::event assignment_ev;
            if (!verbose) {
                /*
                assignment_fixed_window_kernel(
                        X_t,
                        centroids_t,
                        centroids_half_l2_norm,
                        assignment_id,
                    )
                */
                assignment_ev =
                    assignment<
                        dataT, indT,
                        preferred_work_group_size_multiple, 
                        centroids_window_width_multiplier
                    >(
                        exec_q,
                        n_samples, n_features, n_clusters, 
                        centroids_window_height, work_group_size,
                        //
                        X_t, this_centroids_t, 
                        centroids_half_l2_norm, 
                        assignment_id
                    );
            }

            sycl::event compute_inertia_ev;
            bool use_uniform_weights = false;
            if (!verbose || !use_uniform_weights) {
                /*
                # Note that we intentionally we pass unit weights instead of
                # sample_weight so that per_sample_inertia will be updated to the
                # (unweighted) squared distance to the nearest centroid.
                compute_inertia_kernel(
                    X_t,
                    dpt.ones_like(sample_weight),
                    centroids_t,
                    assignments_idx,
                    sq_dist_to_nearest_centroid,
                )
                */
                compute_inertia_ev = 
                    compute_uniform_weight_inertia_kernel<dataT>(
                        exec_q,
                        n_samples, n_features, n_clusters, work_group_size,
                        // 
                        X_t,
                        this_centroids_t, 
                        assignment_id,
                        sq_distance_to_nearest_centroid,
                        {assignment_ev}
                    );
            }

            /*
            self._relocate_empty_clusters(
                    n_empty_clusters_,
                    X_t,
                    sample_weight,
                    new_centroids_t,
                    cluster_sizes,
                    assignments_idx,
                    empty_clusters_list,
                    sq_dist_to_nearest_centroid,
                    per_sample_inertia,
                    work_group_size,
                    compute_dtype,
                )
            */
            relocate_empty_clusters_ev = 
                relocate_empty_clusters<dataT, indT>(
                    exec_q,
                    n_samples, n_features, n_clusters,
                    work_group_size,
                    //
                    host_n_empty_clusters,
                    X_t,                             // IN (n_features, n_samples)
                    sample_weight,                   // IN (n_samples)
                    assignment_id,                   // IN (n_samples, )
                    empty_clusters_list,             // IN (n_clusters, )
                    sq_distance_to_nearest_centroid, // IN (n_samples, )
                    new_centroids_t,                 // INOUT (n_features, n_clusters)
                    cluster_sizes,                   // INOUT (n_clusters,)
                    per_sample_inertia,              // INOUT (n_sample, )
                    {assignment_ev, compute_inertia_ev}
                );
        }

        // compute new_centroids_t /= cluster_sizes
        // broadcast_division_kernel(n_feature, n_clusters, new_centroids_t, cluster_sizes)

        sycl::event broadcast_division_ev =
            broadcast_division_kernel<dataT>(
                exec_q,
                n_features, n_clusters, work_group_size,
                // 
                new_centroids_t, 
                cluster_sizes,
                {relocate_empty_clusters_ev}
            );

        // centroid_shifts = np.square(new_centroids_t - centroids_t).sum(axis=0)
        // compute_centroid_shifts_kernel(
        //     centroids_t, new_centroids_t, centroid_shifts
        // )
        sycl::event compute_centroid_shifts_ev = 
            compute_centroid_shifts_squared_kernel<dataT>(
                exec_q,
                n_features, n_clusters, work_group_size,
                //
                this_centroids_t,     // IN
                new_centroids_t, // IN
                centroid_shifts, // OUT 
                {broadcast_division_ev}
            );

        // centroid_shifts_sum, *_ = reduce_centroid_shifts_kernel(centroid_shifts)
        centroid_shifts_sum = reduce_vector_kernel_blocking<dataT>(
            exec_q,
            n_clusters,
            centroid_shifts,
            {compute_centroid_shifts_ev}
        );

        // centroids_t, new_centroids_t = (new_centroids_t, centroids_t)
        std::swap(this_centroids_t, new_centroids_t);

        ++n_iterations;
    }

    // # Finally, run an assignment kernel to compute the assignments to the best
    // # centroids found, along with the exact inertia.
    // half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

    sycl::event final_half_l2_norm_ev = 
        half_l2_norm_kernel<dataT>(
            exec_q,
            n_features, n_clusters, work_group_size,
            //
            this_centroids_t, 
            centroids_half_l2_norm);

    // assignment_fixed_window_kernel(
    //     X_t, centroids_t, centroids_half_l2_norm, assignments_idx
    // )

    sycl::event final_assignment_ev =
        assignment<
            dataT, indT,
            preferred_work_group_size_multiple, 
            centroids_window_width_multiplier
        >(
            exec_q,
            n_samples, n_features, n_clusters, 
            centroids_window_height, work_group_size,
            //
            X_t, this_centroids_t, 
            centroids_half_l2_norm, 
            assignment_id,
            {final_half_l2_norm_ev}
        );


    // compute_inertia_kernel(
    //     X_t, sample_weight, centroids_t, assignments_idx, per_sample_inertia
    // )
    sycl::event final_compute_inertia_ev = 
        compute_inertia_kernel<dataT>(
            exec_q,
            n_samples, n_features, n_clusters, work_group_size,
            //
            X_t, sample_weight,
            this_centroids_t,
            assignment_id,
            per_sample_inertia,
            {final_assignment_ev}
        );

    sycl::event final_copy_ev;
    if (this_centroids_t != res_centroids_t) {
        final_copy_ev = exec_q.copy<dataT>(this_centroids_t, res_centroids_t, n_features * n_clusters);
    }

    // inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
    // inertia = inertia[0]

    total_inertia =
        reduce_vector_kernel_blocking<dataT>(
            exec_q,
            n_samples,
            per_sample_inertia,
            {final_compute_inertia_ev} 
        );

    final_copy_ev.wait();

    sycl::free(centroids_half_l2_norm, alloc_ctx);
    sycl::free(cluster_sizes, alloc_ctx);
    sycl::free(centroid_shifts, alloc_ctx);
    sycl::free(per_sample_inertia, alloc_ctx);
    sycl::free(new_centroids_t_private_copies, alloc_ctx);
    sycl::free(cluster_sizes_private_copies, alloc_ctx);
    sycl::free(empty_clusters_list, alloc_ctx);

    return n_iterations;
}
