// compute_inertia.hpp

#pragma once
#include <CL/sycl.hpp>
#include <vector>
#include "quotients_utils.hpp"

template <typename T>
class compute_interia_krn;

template <typename T>
sycl::event
compute_inertia_kernel(
    sycl::queue q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    size_t work_group_size,
    // ======================
    const T *X_t,                      // (n_features, n_samples)
    const T *sample_weights,           // (n_features, )
    const T *centroids_t,              // (n_features, n_clusters)
    const size_t *assignments_idx,     // (n_samples, )
    T *per_sample_inertia,             // (n_samples, )
    const std::vector<sycl::event> &depends={}
) {
    sycl::event e = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            auto G = sycl::range<1>(quotient_ceil(n_samples, work_group_size) * work_group_size);
            auto L = sycl::range<1>(work_group_size);

            cgh.parallel_for<class compute_interia_krn<T>>(
                sycl::nd_range<1>(G, L),
                [=](sycl::nd_item<1> it) {
                    size_t sample_idx = it.get_global_id(0);
                    if (sample_idx < n_samples) {
                        T inertia(0);
                        size_t centroid_idx = centroid_idx = assignments_idx[sample_idx];
                        for(size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
                            T diff = X_t[feature_idx * n_samples + sample_idx] - 
                                        centroids_t[feature_idx * n_clusters + centroid_idx];
                            inertia += diff * diff;
                        }
                        per_sample_inertia[sample_idx] = inertia * sample_weights[sample_idx];
                    }
                }
            );
        });

    return e;
}

template <typename T>
class compute_uniform_weight_interia_krn;

template <typename T>
sycl::event
compute_uniform_weight_inertia_kernel(
    sycl::queue q,
    size_t n_samples,
    size_t n_features,
    size_t n_clusters,
    size_t work_group_size,
    // ======================
    const T *X_t,                      // (n_features, n_samples)
    const T *centroids_t,              // (n_features, n_clusters)
    const size_t *assignments_idx,     // (n_samples, )
    T *per_sample_inertia,             // (n_samples, )
    const std::vector<sycl::event> &depends={}
) {
    sycl::event e = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            auto G = sycl::range<1>(quotient_ceil(n_samples, work_group_size) * work_group_size);
            auto L = sycl::range<1>(work_group_size);

            cgh.parallel_for<class compute_uniform_weight_interia_krn<T>>(
                sycl::nd_range<1>(G, L),
                [=](sycl::nd_item<1> it) {
                    size_t sample_idx = it.get_global_id(0);
                    if (sample_idx < n_samples) {
                        T inertia(0);
                        size_t centroid_idx = centroid_idx = assignments_idx[sample_idx];
                        for(size_t feature_idx = 0; feature_idx < n_features; ++feature_idx) {
                            T diff = X_t[feature_idx * n_samples + sample_idx] - 
                                        centroids_t[feature_idx * n_clusters + centroid_idx];
                            inertia += diff * diff;
                        }
                        per_sample_inertia[sample_idx] = inertia;
                    }
                }
            );
        });

    return e;
}


template <typename T>
T reduce_vector_kernel_blocking(
    sycl::queue q,
    size_t n_samples,
    T *data,
    const std::vector<sycl::event> &depends = {}
) {
    T *dev_total = sycl::malloc_device<T>(1, q);

    sycl::event red_ev =
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            sycl::property_list prop( {sycl::property::reduction::initialize_to_identity{}} );
            auto sumReduction = sycl::reduction(dev_total, sycl::plus<T>(), prop);

            cgh.parallel_for(
                {n_samples}, 
                sumReduction,
                [=](sycl::id<1> idx, auto &sum) {
                    sum += data[idx];
                });
        });

    T host_total(0);
    sycl::event copy_ev = 
        q.copy<T>(dev_total, &host_total, 1, {red_ev});

    copy_ev.wait();
    sycl::free(dev_total, q);

    return host_total;
}