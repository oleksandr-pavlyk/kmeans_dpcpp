// device functions

#pragma once

#include <CL/sycl.hpp>
#include <vector>
#include <limits>
#include <utility>

template <typename T, typename slmT>
void _load_window_of_centroids_and_features(
    size_t n_clusters,
    size_t n_features,
    size_t work_group_size, 
    size_t window_n_centroids,
    size_t window_n_features,
    // =====================================
    size_t first_feature_idx,
    size_t loading_centroid_idx,
    size_t window_loading_centroid_idx,
    size_t window_loading_feature_offset,
    T const *current_centroids_t,
    slmT centroids_window
) {
    constexpr T zero(0);

    size_t n_window_features_per_work_group = work_group_size / window_n_centroids;
    size_t centroids_window_height_ratio_multiplier = (
        window_n_features / n_window_features_per_work_group
    );

    size_t centroid_window_first_loading_feature_idx = 0;

    for(size_t i2 = 0; i2 < centroids_window_height_ratio_multiplier; ++i2) {
        size_t window_loading_feature_idx = (
            centroid_window_first_loading_feature_idx + window_loading_feature_offset
        );

        size_t loading_feature_idx = first_feature_idx + window_loading_feature_idx;

        bool in_bound = (loading_feature_idx < n_features && loading_centroid_idx < n_clusters);
        T value = (
            (in_bound) 
            ? current_centroids_t[loading_feature_idx * n_clusters + loading_centroid_idx] 
            : zero
        );

        auto cw_id = sycl::id<2>(window_loading_feature_idx, window_loading_centroid_idx);
        centroids_window[cw_id] = value;
        centroid_window_first_loading_feature_idx += 
            n_window_features_per_work_group;
    }

    return;
}

template <typename T, typename resT>
void _initialize_results(
    size_t n_clusters,
    size_t n_features,
    size_t work_group_size, 
    size_t window_n_centroids,
    size_t window_n_features,
    // ===========================
    resT results
) {
    constexpr T zero(0);
    for(size_t i = 0; i < window_n_centroids; ++i) {
        results[i] = zero;
    }

    return;
}

template <typename T, typename wcT, typename resT>
void _initialize_window_of_centroids(
    size_t n_clusters,
    size_t n_features,
    size_t work_group_size, 
    size_t window_n_centroids,
    size_t window_n_features,
    // ================================
    size_t local_work_id,
    size_t first_centroid_idx,
    const T *centroids_half_l2_norm,
    wcT window_of_centroids_half_l2_norm,
    resT results
) {
    _initialize_results<T>(
        n_clusters, n_features, work_group_size, window_n_centroids, window_n_features,
        results
    );

    constexpr T inf = std::numeric_limits<T>::infinity();

    // The first `window_n_centroids` work items cooperate on loading the
    // values of centroids_half_l2_norm relevant to current window. Each work
    // item loads one single value.
    size_t half_l2_norm_loading_idx = first_centroid_idx + local_work_id;
    if (local_work_id < window_n_centroids) {
        bool in_bound = half_l2_norm_loading_idx < n_clusters;
        T l2_norm = (in_bound) ? centroids_half_l2_norm[half_l2_norm_loading_idx] : inf;
        window_of_centroids_half_l2_norm[local_work_id] = l2_norm;
    }
}

template <typename T, typename cwT, typename resT, bool acummulate_dot_product>
void _acummulate_sum_of_ops(
    size_t n_samples, 
    size_t n_features, 
    size_t window_n_features, 
    size_t window_n_centroids,
    // ===========================
    size_t sample_idx,
    size_t first_feature_idx,
    const T *X_t,
    cwT centroids_window,
    resT result
) {
    constexpr T zero(0);
    bool in_bound_sample = (sample_idx < n_samples);
    for(size_t window_feature_idx = 0; window_feature_idx < window_n_features; ++window_feature_idx) {
        size_t feature_idx = window_feature_idx + first_feature_idx;

        bool in_bound = in_bound_sample && (feature_idx < n_features);
        T X_value = (in_bound) ? X_t[feature_idx * n_samples + sample_idx] : zero;

        for(size_t window_centroid_idx = 0; window_centroid_idx < window_n_centroids; ++window_centroid_idx) {
            T centroid_value = centroids_window[sycl::id<2>(window_feature_idx, window_centroid_idx)];
            if constexpr (acummulate_dot_product) {
                result[window_centroid_idx] += centroid_value * X_value;
            } else {
                T diff = centroid_value - X_value;
                result[window_centroid_idx] += diff * diff;
            }
        }
    }
}

template <typename T>
std::pair<size_t, T> _update_closest_centroid(
    size_t window_n_centroids,
    // =================================
    size_t first_centroid_idx,
    size_t min_idx,
    T min_sample_pseudo_inertia,
    const T *window_of_centroids_half_l2_norms,
    const T *dot_products
) {
    size_t min_idx_ = min_idx;
    T min_sample_pseudo_inertia_ = min_sample_pseudo_inertia;

    for(size_t i = 0; i < window_n_centroids; ++i) {
        T current_sample_pseudo_inertia = 
             window_of_centroids_half_l2_norms[i] - dot_products[i];

        bool update = (current_sample_pseudo_inertia < min_sample_pseudo_inertia_);
        min_idx_ = (update) ? first_centroid_idx + i : min_idx_;
        min_sample_pseudo_inertia_ = (update) ? current_sample_pseudo_inertia : min_sample_pseudo_inertia_;
    }

    return std::make_pair<size_t, T>(min_idx_, min_sample_pseudo_inertia_);
}

