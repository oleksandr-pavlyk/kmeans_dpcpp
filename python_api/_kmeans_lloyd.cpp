#include <cstdint>
#include <vector>
#include <utility>
#include <sstream>
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"

#include "util_kernels.hpp"
#include "compute_euclidean_distance.hpp"
#include "assignment.hpp"
#include "compute_inertia.hpp"
#include "lloyd_single_step.hpp"
#include "kmeans_lloyd_driver.hpp"

namespace py = pybind11;

constexpr size_t preferred_work_group_size_multiple = 8;
constexpr size_t centroids_window_width_multiplier = 4;

template <std::size_t num>
bool all_c_contiguous(const dpctl::tensor::usm_ndarray (&args)[num]) {
  bool all_contig = true;
  for (size_t i = 0; all_contig && i < num; ++i) {
    all_contig = args[i].is_c_contiguous();
  }
  return all_contig;
}

template <std::size_t num>
bool same_typenum_as(int typenum, const dpctl::tensor::usm_ndarray (&args)[num]) {
  bool res = true;
  for(size_t i=0; res && i < num; ++i) {
    res = (typenum == args[i].get_typenum());
  }
  return res;
}

bool is_1d(const dpctl::tensor::usm_ndarray &ar) { return (1 == ar.get_ndim()); }
bool is_2d(const dpctl::tensor::usm_ndarray &ar) { return (2 == ar.get_ndim()); }
bool is_3d(const dpctl::tensor::usm_ndarray &ar) { return (3 == ar.get_ndim()); }
bool is_0d(const dpctl::tensor::usm_ndarray &ar) { return 1 == ar.get_size(); }

template <std::size_t num>
bool queues_are_compatible(sycl::queue exec_q,
			   const ::dpctl::tensor::usm_ndarray (&arrs)[num])
{
    for (std::size_t i = 0; i < num; ++i) {
        if (exec_q != arrs[i].get_queue()) {
            return false;
        }
    }
    return true;
}


/*! @brief Evaluates X /= y */
std::pair<sycl::event, sycl::event>
py_broadcast_divide(
  dpctl::tensor::usm_ndarray X,
  dpctl::tensor::usm_ndarray y,
  sycl::queue q,
  const std::vector<sycl::event> &depends={})
{

  if (!is_2d(X) || !is_1d(y) || !all_c_contiguous({X, y})) {
    throw py::value_error("Arguments must be a matrix and a vector with C-contiguous layout");
  }

  if (y.get_shape(0) != X.get_shape(1)) {
    throw py::value_error("Array dimensions of arguments are not consistent, X.shape[1] != y.shape[0]");
  }

  if (!queues_are_compatible(q, {X, y})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto y_typenum = y.get_typenum();
  if (!same_typenum_as(y_typenum, {X})) {
    throw py::value_error("Arguments must have the same elemental data types.");
  }

  auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (y_typenum == api.UAR_FLOAT_) {
    comp_ev = broadcast_division_kernel<float>(
              q, X.get_shape(0), X.get_shape(1), 32, X.get_data<float>(), y.get_data<float>(), depends
           );
  } else if (y_typenum == api.UAR_DOUBLE_) {
    comp_ev = broadcast_division_kernel<double>(
              q, X.get_shape(0), X.get_shape(1), 32, X.get_data<double>(), y.get_data<double>(), depends
           );
  } else {
    throw py::value_error("Unsupported elemental data type. Expecting single or double precision floating point numbers");
  }

  sycl::event ht_ev = ::dpctl::utils::keep_args_alive(q, {X, y}, {comp_ev});
  return std::make_pair(ht_ev, comp_ev);
}

/*! @brief Evaluates y = np.square(np.linalg.norm(X, axis=1)) / 2 */
std::pair<sycl::event, sycl::event>
py_half_l2_norm_squared(
  dpctl::tensor::usm_ndarray X,
  dpctl::tensor::usm_ndarray y,
  sycl::queue q,
  const std::vector<sycl::event> &depends={})
{
  if (!is_2d(X) || !is_1d(y) || !all_c_contiguous({X, y})) {
    throw py::value_error("Arguments must be a matrix and a vector with C-contiguous layout");
  }

  if (y.get_shape(0) != X.get_shape(1)) {
    throw py::value_error("Array dimensions of arguments are not consistent, X.shape[1] != y.shape[0]");
  }

  if (!queues_are_compatible(q, {X, y})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto y_typenum = y.get_typenum();
  if (!same_typenum_as(y_typenum, {X})) {
    throw py::value_error("Arguments must have the same elemental data types.");
  }

  const auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (y_typenum == api.UAR_FLOAT_) {
    comp_ev = half_l2_norm_kernel<float>(
              q, X.get_shape(0), X.get_shape(1), 32, X.get_data<float>(), y.get_data<float>(), depends
           );
  } else if (y_typenum == api.UAR_DOUBLE_) {
    comp_ev = half_l2_norm_kernel<double>(
              q, X.get_shape(0), X.get_shape(1), 32, X.get_data<double>(), y.get_data<double>(), depends
           );
  } else {
    throw py::value_error("Unsupported elemental data type. Expecting single or double precision floating point numbers");
  }

  sycl::event ht_ev = ::dpctl::utils::keep_args_alive(q, {X, y}, {comp_ev});
  return std::make_pair(ht_ev, comp_ev);
}


std::pair<sycl::event, sycl::event>
py_reduce_centroids_data(
  dpctl::tensor::usm_ndarray cluster_sizes_private_copies, // IN (n_copies, n_clusters)                dataT
  dpctl::tensor::usm_ndarray centroids_t_private_copies,   // IN (n_copies, n_features, n_clusters,)   dataT
  dpctl::tensor::usm_ndarray out_cluster_sizes,            // OUT (n_clusters,)                        dataT
  dpctl::tensor::usm_ndarray out_centroids_t,              // OUT (n_features, n_clusters,)            dataT
  dpctl::tensor::usm_ndarray out_empty_clusters_list,      // OUT (n_clusters,)                        indT
  dpctl::tensor::usm_ndarray out_n_empty_clusters,         // OUT (1,)                                 indT
  size_t work_group_size,
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {
  if (!is_2d(cluster_sizes_private_copies) ||
      !is_3d(centroids_t_private_copies) ||
      !is_1d(out_cluster_sizes) ||
      !is_2d(out_centroids_t) ||
      !is_1d(out_empty_clusters_list) ||
      !is_1d(out_n_empty_clusters))
  {
    throw py::value_error("Array dimensions of inputs are not consistent");
  }

  py::ssize_t n_copies = cluster_sizes_private_copies.get_shape(0);
  py::ssize_t n_clusters = cluster_sizes_private_copies.get_shape(1);
  py::ssize_t n_features = centroids_t_private_copies.get_shape(1);

  if ( n_copies != centroids_t_private_copies.get_shape(0) ||
       n_clusters != centroids_t_private_copies.get_shape(2) ||
       n_clusters != out_cluster_sizes.get_shape(0) ||
       n_clusters != out_centroids_t.get_shape(1) ||
       n_features != out_centroids_t.get_shape(0) ||
       n_clusters != out_empty_clusters_list.get_shape(0) )
  {
    throw py::value_error("Dimensions mismatch");
  }

  int dataT_typenum = cluster_sizes_private_copies.get_typenum();
  int indT_typenum = out_n_empty_clusters.get_typenum();

  if (!same_typenum_as(dataT_typenum, {centroids_t_private_copies, out_cluster_sizes, out_centroids_t}) ||
      !same_typenum_as(indT_typenum, {out_empty_clusters_list}))
  {
    throw py::value_error("Array element data types must be consisten");
  }

  if (!all_c_contiguous({
        cluster_sizes_private_copies, centroids_t_private_copies,
        out_cluster_sizes, out_centroids_t, out_empty_clusters_list,
        out_n_empty_clusters
      }))
  {
    throw py::value_error("All array arguments must be C-contiguous");
  }

  if (! queues_are_compatible(q, {
				  cluster_sizes_private_copies,
				  centroids_t_private_copies,
				  out_cluster_sizes,
				  out_centroids_t,
				  out_empty_clusters_list,
				  out_n_empty_clusters
      })) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto &api = ::dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;

  if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;
    comp_ev = reduce_centroid_data_kernel<dataT, indT>(
            q, n_copies, n_features, n_clusters, work_group_size,
            cluster_sizes_private_copies.get_data<dataT>(),
            centroids_t_private_copies.get_data<dataT>(),
            out_cluster_sizes.get_data<dataT>(),
            out_centroids_t.get_data<dataT>(),
            out_empty_clusters_list.get_data<indT>(),
            out_n_empty_clusters.get_data<indT>(),
            depends);
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;
    comp_ev = reduce_centroid_data_kernel<dataT, indT>(
            q, n_copies, n_features, n_clusters, work_group_size,
            cluster_sizes_private_copies.get_data<dataT>(),
            centroids_t_private_copies.get_data<dataT>(),
            out_cluster_sizes.get_data<dataT>(),
            out_centroids_t.get_data<dataT>(),
            out_empty_clusters_list.get_data<indT>(),
            out_n_empty_clusters.get_data<indT>(),
            depends);
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;
    comp_ev = reduce_centroid_data_kernel<dataT, indT>(
            q, n_copies, n_features, n_clusters, work_group_size,
            cluster_sizes_private_copies.get_data<dataT>(),
            centroids_t_private_copies.get_data<dataT>(),
            out_cluster_sizes.get_data<dataT>(),
            out_centroids_t.get_data<dataT>(),
            out_empty_clusters_list.get_data<indT>(),
            out_n_empty_clusters.get_data<indT>(),
            depends);
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;
    comp_ev = reduce_centroid_data_kernel<dataT, indT>(
            q, n_copies, n_features, n_clusters, work_group_size,
            cluster_sizes_private_copies.get_data<dataT>(),
            centroids_t_private_copies.get_data<dataT>(),
            out_cluster_sizes.get_data<dataT>(),
            out_centroids_t.get_data<dataT>(),
            out_empty_clusters_list.get_data<indT>(),
            out_n_empty_clusters.get_data<indT>(),
            depends);
  } else {
    throw py::value_error("Unsupported data types");
  }

  sycl::event ht_ev = ::dpctl::utils::keep_args_alive(
       q, {cluster_sizes_private_copies, centroids_t_private_copies,
           out_cluster_sizes, out_centroids_t, out_empty_clusters_list,
           out_n_empty_clusters}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_compute_threshold(
  dpctl::tensor::usm_ndarray data,
  size_t topk,
  dpctl::tensor::usm_ndarray threshold,
  sycl::queue q,
  const std::vector<sycl::event> &depends = {}
) {
  if (!is_1d(data) || !all_c_contiguous({data, threshold})) {
    throw py::value_error("Argument data must be a C-contiguous vector");
  }

  if (!is_0d(threshold)) {
    throw py::value_error("Argument threshold must be 1-element array");
  }

  int data_typenum = data.get_typenum();
  if(!same_typenum_as(data_typenum, {threshold})) {
    throw py::value_error("Data types of arguments must be the same");
  }

  if (! queues_are_compatible(q, {data, threshold})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto &api = ::dpctl::detail::dpctl_capi::get();
  py::ssize_t n_samples = data.get_shape(0);

  if (topk == 0) {
    sycl::event e1{};
    sycl::event e2{};
    return std::make_pair(e1, e2);
  }

  sycl::event comp_ev;
  if (data_typenum == api.UAR_FLOAT_) {
    using dataT = float;
    comp_ev = compute_threshold_kernel<dataT>(q,
      n_samples, data.get_data<dataT>(), topk,
      threshold.get_data<dataT>(), depends);
  } else if (data_typenum == api.UAR_DOUBLE_) {
    using dataT = double;
    comp_ev = compute_threshold_kernel<dataT>(q,
      n_samples, data.get_data<dataT>(), topk,
      threshold.get_data<dataT>(), depends);
  } else {
    throw py::value_error("Unsupported elemental data type. Expect single- or double- floating-point types.");
  }

  sycl::event ht_ev = ::dpctl::utils::keep_args_alive(
       q, {data, threshold}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_select_samples_far_from_centroid(
  size_t n_selected,
  dpctl::tensor::usm_ndarray distance_to_centroid,  // IN (n_samples, )    dataT
  dpctl::tensor::usm_ndarray threshold,             // IN (1,)             dataT
  dpctl::tensor::usm_ndarray selected_samples_idx,  // OUT (n_sample, )    indT
  dpctl::tensor::usm_ndarray n_selected_gt_threshold, // OUT (1,)          indT
  dpctl::tensor::usm_ndarray n_selected_eq_threshold, // OUT (1,)          indT
  size_t work_group_size,
  sycl::queue q,
  const std::vector<sycl::event> &depends = {}
) {

  if (n_selected == 0) { throw py::value_error("Argument `n_selected` must be positive");}

  if (!is_1d(distance_to_centroid) || !is_1d(selected_samples_idx) ||
      !is_0d(threshold) || !is_0d(n_selected_gt_threshold) ||
      !is_0d(n_selected_eq_threshold)
  ) {
    throw py::value_error("Array dimensionalities are not consistent");
  }

  py::ssize_t n_samples = distance_to_centroid.get_shape(0);
  if ( static_cast<size_t>(n_samples) < n_selected) {
    throw py::value_error("Argument `n_select` is too large");
  }

  if (n_samples != selected_samples_idx.get_shape(0)) {
    throw py::value_error("Vector `selected_samples_idx` must have size `n_samples`");
  }

  if (!all_c_contiguous({distance_to_centroid, selected_samples_idx})) {
    throw py::value_error("Arrays must be C-contiguous");
  }

  int dataT_typenum = distance_to_centroid.get_typenum();
  if (!same_typenum_as(dataT_typenum, {threshold})) {
    throw py::value_error("Input arrays must have the same elemental data type");
  }

  int indT_typenum = selected_samples_idx.get_typenum();

  if (!same_typenum_as(indT_typenum, {n_selected_gt_threshold, n_selected_eq_threshold}))
  {
      throw py::value_error("Output array must have the same elemental data type");
  }

  if (!queues_are_compatible(q, {
    distance_to_centroid,
    threshold, selected_samples_idx,
    n_selected_gt_threshold,
    n_selected_eq_threshold
  })) { throw py::value_error("Execution queue is not compatible with allocation queues"); }

  auto &api = ::dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;
    comp_ev = select_samples_far_from_centroid_kernel<dataT, indT>(
      q, n_selected, static_cast<size_t>(n_samples), work_group_size,
      distance_to_centroid.get_data<dataT>(), threshold.get_data<dataT>(),
      selected_samples_idx.get_data<indT>(), n_selected_gt_threshold.get_data<indT>(),
      n_selected_eq_threshold.get_data<indT>(), depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;
    comp_ev = select_samples_far_from_centroid_kernel<dataT, indT>(
      q, n_selected, static_cast<size_t>(n_samples), work_group_size,
      distance_to_centroid.get_data<dataT>(), threshold.get_data<dataT>(),
      selected_samples_idx.get_data<indT>(), n_selected_gt_threshold.get_data<indT>(),
      n_selected_eq_threshold.get_data<indT>(), depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;
    comp_ev = select_samples_far_from_centroid_kernel<dataT, indT>(
      q, n_selected, static_cast<size_t>(n_samples), work_group_size,
      distance_to_centroid.get_data<dataT>(), threshold.get_data<dataT>(),
      selected_samples_idx.get_data<indT>(), n_selected_gt_threshold.get_data<indT>(),
      n_selected_eq_threshold.get_data<indT>(), depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;
    comp_ev = select_samples_far_from_centroid_kernel<dataT, indT>(
      q, n_selected, static_cast<size_t>(n_samples), work_group_size,
      distance_to_centroid.get_data<dataT>(), threshold.get_data<dataT>(),
      selected_samples_idx.get_data<indT>(), n_selected_gt_threshold.get_data<indT>(),
      n_selected_eq_threshold.get_data<indT>(), depends
    );
  } else {
    throw py::value_error("Unsupported data types");
  }

  sycl::event ht_ev = ::dpctl::utils::keep_args_alive(q, {
    distance_to_centroid, threshold, selected_samples_idx,
    n_selected_gt_threshold, n_selected_eq_threshold}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_relocate_empty_clusters(
  size_t n_empty_clusters,
  dpctl::tensor::usm_ndarray X_t,             // IN (n_features, n_samples, )  dataT
  dpctl::tensor::usm_ndarray sample_weights,  // IN (n_samples, )              dataT
  dpctl::tensor::usm_ndarray assignment_id,   // IN (n_samples, )              indT
  dpctl::tensor::usm_ndarray empty_clusters_list, // IN (n_clusters)           indT
  dpctl::tensor::usm_ndarray sq_dist_to_nearest_centroid, // IN (n_samples,)   dataT
  dpctl::tensor::usm_ndarray centroid_t,      // IN-OUT  (n_features, n_clusters, )  dataT
  dpctl::tensor::usm_ndarray cluster_sizes,   // IN-OUT  (n_clusters, )        dataT
  dpctl::tensor::usm_ndarray per_sample_inertia, // IN-OUT (n_samples, )       dataT
  size_t work_group_size,
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
)
{
  if ( !is_2d(X_t) || !is_1d(sample_weights) ||
       !is_1d(assignment_id) ||
       !is_1d(empty_clusters_list) ||
       !is_1d(sq_dist_to_nearest_centroid) ||
       !is_2d(centroid_t) ||
       !is_1d(cluster_sizes) ||
       !is_1d(per_sample_inertia)
      )
  {
    throw py::value_error("Arguments have inconsistent array dimensionality.");
  }

  if (!all_c_contiguous({X_t, sample_weights, assignment_id, empty_clusters_list,
                        sq_dist_to_nearest_centroid, centroid_t, cluster_sizes, per_sample_inertia}))
  {
    throw py::value_error("Inputs must be C-contiguous");
  }

  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_clusters = empty_clusters_list.get_shape(0);

  if (n_samples != sample_weights.get_shape(0)              ||
      n_samples != assignment_id.get_shape(0)               ||
      n_samples != sq_dist_to_nearest_centroid.get_shape(0) ||
      n_clusters != centroid_t.get_shape(1)                 ||
      n_features != centroid_t.get_shape(0)                 ||
      n_clusters != cluster_sizes.get_shape(0)              ||
      n_samples != per_sample_inertia.get_shape(0)
  )
  {
    throw py::value_error("Input dimensions are inconsistent");
  }

  if (n_empty_clusters == 0) {
    throw py::value_error("n_empty_clusters must be non-zero.");
  }

  if (!queues_are_compatible(q, {
        X_t, sample_weights,
        assignment_id, empty_clusters_list,
        sq_dist_to_nearest_centroid, centroid_t,
        cluster_sizes, per_sample_inertia
        }))
  {
    throw py::value_error("Execution queue is not compatible with allocation queues.");
  }

  int dataT_typenum = X_t.get_typenum();
  int indT_typenum = assignment_id.get_typenum();

  if (!same_typenum_as(dataT_typenum, {sample_weights, sq_dist_to_nearest_centroid, centroid_t, cluster_sizes, per_sample_inertia}) ||
      !same_typenum_as(indT_typenum, {empty_clusters_list}))
  {
    throw py::value_error("Inconsistent array elemental data types");
  }

  const auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    comp_ev = relocate_empty_clusters<dataT, indT>(
      q,
      n_samples, n_features, static_cast<indT>(n_clusters), work_group_size,
      n_empty_clusters, X_t.get_data<dataT>(), sample_weights.get_data<dataT>(),
      assignment_id.get_data<indT>(), empty_clusters_list.get_data<indT>(),
      sq_dist_to_nearest_centroid.get_data<dataT>(),
      centroid_t.get_data<dataT>(), cluster_sizes.get_data<dataT>(),
      per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    comp_ev = relocate_empty_clusters<dataT, indT>(
      q,
      n_samples, n_features, static_cast<indT>(n_clusters), work_group_size,
      n_empty_clusters, X_t.get_data<dataT>(), sample_weights.get_data<dataT>(),
      assignment_id.get_data<indT>(), empty_clusters_list.get_data<indT>(),
      sq_dist_to_nearest_centroid.get_data<dataT>(),
      centroid_t.get_data<dataT>(), cluster_sizes.get_data<dataT>(),
      per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;

    comp_ev = relocate_empty_clusters<dataT, indT>(
      q,
      n_samples, n_features, static_cast<indT>(n_clusters), work_group_size,
      n_empty_clusters, X_t.get_data<dataT>(), sample_weights.get_data<dataT>(),
      assignment_id.get_data<indT>(), empty_clusters_list.get_data<indT>(),
      sq_dist_to_nearest_centroid.get_data<dataT>(),
      centroid_t.get_data<dataT>(), cluster_sizes.get_data<dataT>(),
      per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;

    comp_ev = relocate_empty_clusters<dataT, indT>(
      q,
      n_samples, n_features, static_cast<indT>(n_clusters), work_group_size,
      n_empty_clusters, X_t.get_data<dataT>(), sample_weights.get_data<dataT>(),
      assignment_id.get_data<indT>(), empty_clusters_list.get_data<indT>(),
      sq_dist_to_nearest_centroid.get_data<dataT>(),
      centroid_t.get_data<dataT>(), cluster_sizes.get_data<dataT>(),
      per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else {
    throw py::value_error("Unsupported data types");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(q,
    {
      X_t, sample_weights, assignment_id, empty_clusters_list,
      sq_dist_to_nearest_centroid, centroid_t, cluster_sizes
    },
    {comp_ev}
  );

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_compute_centroid_shifts_squared_kernel(
  dpctl::tensor::usm_ndarray old_centroid_t,    // IN (n_features, n_clusters)
  dpctl::tensor::usm_ndarray new_centroid_t,    // IN (n_features, n_clusters)
  dpctl::tensor::usm_ndarray centroid_shifts,   // OUT (n_clusters)
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {
  if (!is_2d(new_centroid_t) ||
      !is_2d(old_centroid_t) ||
      !is_1d(centroid_shifts)
  ) {
    throw py::value_error("Input dimensionalities are not consistent.");
  }

  py::ssize_t n_features = old_centroid_t.get_shape(0);
  py::ssize_t n_clusters = old_centroid_t.get_shape(1);

  if (n_features != new_centroid_t.get_shape(0) ||
      n_clusters != new_centroid_t.get_shape(1) ||
      n_clusters != centroid_shifts.get_shape(0)
  ) {
    throw py::value_error("Array dimensions are not consistent.");
  }

  if (!all_c_contiguous({new_centroid_t, old_centroid_t, centroid_shifts})) {
    throw py::value_error("Arguments must be C-contiguous arrays");
  }

  if (!queues_are_compatible(q, {new_centroid_t, old_centroid_t, centroid_shifts})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  int typenum = old_centroid_t.get_typenum();
  if (!same_typenum_as(typenum, {new_centroid_t, centroid_shifts})) {
    throw py::value_error("All array arguments must have the same elemental data types");
  }

  auto &api = ::dpctl::detail::dpctl_capi::get();
  size_t work_group_size = 128;

  sycl::event comp_ev;
  if (typenum == api.UAR_FLOAT_ ) {
    using dataT = float;

    comp_ev = compute_centroid_shifts_squared_kernel<dataT>(
      q, n_features, n_clusters, work_group_size,
      old_centroid_t.get_data<dataT>(),
      new_centroid_t.get_data<dataT>(),
      centroid_shifts.get_data<dataT>(),
      depends
    );
  } else if (typenum == api.UAR_DOUBLE_) {
    using dataT = double;

    comp_ev = compute_centroid_shifts_squared_kernel<dataT>(
      q, n_features, n_clusters, work_group_size,
      old_centroid_t.get_data<dataT>(),
      new_centroid_t.get_data<dataT>(),
      centroid_shifts.get_data<dataT>(),
      depends
    );

  } else {
    throw py::value_error("Unsupported elemental data type.");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(
    q,
    {old_centroid_t, new_centroid_t, centroid_shifts}, {comp_ev}
  );

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_compute_distances(
  dpctl::tensor::usm_ndarray X_t,                    // IN (n_feautes, n_samples)
  dpctl::tensor::usm_ndarray centroid_t,             // IN (n_features, n_clusters)
  dpctl::tensor::usm_ndarray euclidean_distances_t,  // OUT (n_clusters, n_samples)
  size_t work_group_size,
  size_t centroids_window_height,
  sycl::queue q,
  const std::vector<sycl::event> &depends = {}
) {
  if ( !is_2d(X_t) || !is_2d(centroid_t) || !is_2d(euclidean_distances_t)) {
    throw py::value_error("Input arrays must have dimensionality 2.");
  }

  if (!all_c_contiguous({X_t, centroid_t, euclidean_distances_t})) {
    throw py::value_error("Input arrays must be C-contiguous.");
  }

  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_clusters = euclidean_distances_t.get_shape(0);

  if ( n_features != centroid_t.get_shape(0) || n_clusters != centroid_t.get_shape(1) || n_samples != euclidean_distances_t.get_shape(1)) {
    std::cout << (n_features != centroid_t.get_shape(0))
    << " " <<  (n_clusters != centroid_t.get_shape(1)) << " " << (n_samples != euclidean_distances_t.get_shape(1)) << std::endl;
    throw py::value_error("Input array dimensions are not consistant");
  }

  if (!queues_are_compatible(q, {X_t, centroid_t, euclidean_distances_t})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  int typenum = X_t.get_typenum();

  if (!same_typenum_as(typenum, {centroid_t, euclidean_distances_t})) {
    throw py::value_error("Arrays must have the same elemental data types");
  }

  const auto &api = ::dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (typenum == api.UAR_FLOAT_) {
    using dataT = float;

    comp_ev = compute_distances<dataT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples,
      n_features,
      n_clusters,
      centroids_window_height,
      work_group_size,
      X_t.get_data<dataT>(),
      centroid_t.get_data<dataT>(),
      euclidean_distances_t.get_data<dataT>(),
      depends
    );
  } else if (typenum == api.UAR_DOUBLE_) {
    using dataT = double;

    comp_ev = compute_distances<dataT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples,
      n_features,
      n_clusters,
      centroids_window_height,
      work_group_size,
      X_t.get_data<dataT>(),
      centroid_t.get_data<dataT>(),
      euclidean_distances_t.get_data<dataT>(),
      depends
    );
  } else {
    throw py::value_error("Unsupported elemental data type");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(q,
    {X_t, centroid_t, euclidean_distances_t}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_assignment(
  dpctl::tensor::usm_ndarray X_t,        // IN (n_features, n_samples)
  dpctl::tensor::usm_ndarray centroid_t, // IN (n_features, n_clusters)
  dpctl::tensor::usm_ndarray centroids_half_l2_norm, // (n_clusters,)
  dpctl::tensor::usm_ndarray assignment_id,  // OUT (n_samples, )
  size_t centroids_window_height,
  size_t work_group_size,
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {
  if ( !is_2d(X_t) || !is_2d(centroid_t) || !is_1d(centroids_half_l2_norm) || !is_1d(assignment_id)) {
    throw py::value_error("Inputs have unexpected dimensionality.");
  }

  if (!all_c_contiguous({X_t, centroid_t, centroids_half_l2_norm, assignment_id})) {
    throw py::value_error("Inputs must be C-contiguous arrays.");
  }

  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_clusters = centroids_half_l2_norm.get_shape(0);

  if (n_features != centroid_t.get_shape(0) || n_clusters != centroid_t.get_shape(1) || n_samples != assignment_id.get_shape(0)) {
    throw py::value_error("Inputs have inconsistent dimensions.");
  }

  if(!queues_are_compatible(q, {X_t, centroid_t, centroids_half_l2_norm, assignment_id})) {
    throw py::value_error("Execution queue is incompatible with allocation queues.");
  }

  int dataT_typenum = X_t.get_typenum();
  int indT_typenum = assignment_id.get_typenum();

  if (!same_typenum_as(dataT_typenum, {centroid_t, centroids_half_l2_norm})) {
    throw py::value_error("Arrays have inconsistent elemental data types");
  }

  const auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if(dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    comp_ev = assignment<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), centroid_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignment_id.get_data<indT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;

    comp_ev = assignment<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), centroid_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignment_id.get_data<indT>(),
      depends
    );
  } else if(dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    comp_ev = assignment<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), centroid_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignment_id.get_data<indT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;

    comp_ev = assignment<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), centroid_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignment_id.get_data<indT>(),
      depends
    );
  } else {
    throw py::value_error("Unsupported array elemental data type");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(q, {X_t, centroid_t, centroids_half_l2_norm, assignment_id}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

std::pair<sycl::event, sycl::event>
py_compute_inertia(
  dpctl::tensor::usm_ndarray X_t,
  dpctl::tensor::usm_ndarray sample_weight,
  dpctl::tensor::usm_ndarray centroid_t,
  dpctl::tensor::usm_ndarray assignment_id,
  dpctl::tensor::usm_ndarray per_sample_inertia,
  size_t work_group_size,
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {

  if ( !is_2d(X_t) || !is_1d(sample_weight) || !is_2d(centroid_t) || !is_1d(assignment_id) || !is_1d(per_sample_inertia)) {
    throw py::value_error("Input array dimensionality is not consistent");
  }

  if (!all_c_contiguous({X_t, sample_weight, centroid_t, assignment_id, per_sample_inertia})) {
    throw py::value_error("All input arrays must be C-contiguous");
  }

  if (!queues_are_compatible(q, {
        X_t, sample_weight, centroid_t,
        assignment_id, per_sample_inertia
      })
  ) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_clusters = centroid_t.get_shape(1);

  if (n_features != centroid_t.get_shape(0) || n_samples != sample_weight.get_shape(0) ||
        n_samples != assignment_id.get_shape(0) || n_samples != per_sample_inertia.get_shape(0)) {
    throw py::value_error("Array dimensions are not consistent");
  }

  int dataT_typenum = X_t.get_typenum();
  int indT_typenum = assignment_id.get_typenum();

  if (!same_typenum_as(dataT_typenum, {sample_weight, centroid_t,  per_sample_inertia})) {
    throw py::value_error("Array elemental data types are not consistent");
  }

  const auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    comp_ev = compute_inertia_kernel<dataT, indT>(
      q,
      n_samples, n_features, n_clusters, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroid_t.get_data<dataT>(),
      assignment_id.get_data<indT>(), per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;

    comp_ev = compute_inertia_kernel<dataT, indT>(
      q,
      n_samples, n_features, n_clusters, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroid_t.get_data<dataT>(),
      assignment_id.get_data<indT>(), per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    comp_ev = compute_inertia_kernel<dataT, indT>(
      q,
      n_samples, n_features, n_clusters, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroid_t.get_data<dataT>(),
      assignment_id.get_data<indT>(), per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;

    comp_ev = compute_inertia_kernel<dataT, indT>(
      q,
      n_samples, n_features, n_clusters, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroid_t.get_data<dataT>(),
      assignment_id.get_data<indT>(), per_sample_inertia.get_data<dataT>(),
      depends
    );
  } else {
    throw py::value_error("Unsupported array elemental data type");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(q, {X_t, sample_weight, centroid_t, assignment_id, per_sample_inertia}, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

py::object
py_reduce_vector_blocking(
  dpctl::tensor::usm_ndarray data,
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {

  if (!is_1d(data) || !all_c_contiguous({data})) {
    throw py::value_error("Expecting 1D contiguous vector.");
  }

  if (!queues_are_compatible(q, {data})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  py::ssize_t n_samples = data.get_shape(0);

  int typenum = data.get_typenum();
  const auto &api = dpctl::detail::dpctl_capi::get();

  if (typenum == api.UAR_FLOAT_) {
    using dataT = float;

    dataT res = reduce_vector_kernel_blocking<dataT>(q, n_samples, data.get_data<dataT>(), depends);
    return py::cast(res);

  } else if (typenum == api.UAR_DOUBLE_) {
    using dataT = double;

    dataT res = reduce_vector_kernel_blocking<dataT>(q, n_samples, data.get_data<dataT>(), depends);
    return py::cast(res);

  } else {
    throw py::value_error("Unsupported data type");
  }
}

/*! @brief Returns pair of events, host-task event keeping argument Python objects alive,
    and event signaling completion of tasks submitted by this routine. */
std::pair<sycl::event, sycl::event>
py_fused_lloyd_single_step(
  dpctl::tensor::usm_ndarray X_t,                             // IN   (n_features, n_samples)
  dpctl::tensor::usm_ndarray sample_weight,                   // IN   (n_samples)
  dpctl::tensor::usm_ndarray centroids_t,                      // IN   (n_features, n_clusters,)
  dpctl::tensor::usm_ndarray centroids_half_l2_norm,          // IN   (n_clusters,)
  dpctl::tensor::usm_ndarray assignments_idx,                 // OUT  (n_samples, )
  dpctl::tensor::usm_ndarray new_centroids_t_private_copies,  // OUT  (n_private_copies, n_features, n_clusters)
  dpctl::tensor::usm_ndarray cluster_sizes_private_copies,    // OUT  (n_private_copies, n_clusters)
  size_t centroids_window_height,                             //
  size_t work_group_size,
  sycl::queue q,                                              // execution queue
  const std::vector<sycl::event> &depends = {}                // task dependencies
) {
  if (!is_2d(X_t) || !is_1d(sample_weight) || !is_2d(centroids_t) ||
         !is_1d(centroids_half_l2_norm) || !is_1d(assignments_idx) ||
            !is_3d(new_centroids_t_private_copies) || !is_2d(cluster_sizes_private_copies))
  {
    throw py::value_error("Unexpected input array dimensionalities.");
  }

  if (!all_c_contiguous({X_t, sample_weight, centroids_t, centroids_half_l2_norm,
           assignments_idx, new_centroids_t_private_copies, cluster_sizes_private_copies}))
  {
    throw py::value_error("All arrays must be C-contiguous");
  }

  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_clusters = centroids_half_l2_norm.get_shape(0);
  py::ssize_t n_copies = new_centroids_t_private_copies.get_shape(0);

  if (n_features != centroids_t.get_shape(0) || n_clusters != centroids_t.get_shape(1) ||
      n_samples != sample_weight.get_shape(0) || n_samples != assignments_idx.get_shape(0) ||
      n_features != new_centroids_t_private_copies.get_shape(1) ||
      n_clusters != new_centroids_t_private_copies.get_shape(2) ||
      n_copies != cluster_sizes_private_copies.get_shape(0) ||
      n_clusters != cluster_sizes_private_copies.get_shape(1)
  ) {
    throw py::value_error("Unexpected array dimensions");
  }

  if (!queues_are_compatible(q, {
    X_t, sample_weight, centroids_t,
    centroids_half_l2_norm, assignments_idx,
    new_centroids_t_private_copies, cluster_sizes_private_copies
  })) {
    throw py::value_error("Execution queue is not compatible with allocation queues.");
  }

  int dataT_typenum = X_t.get_typenum();
  int indT_typenum = assignments_idx.get_typenum();

  if (!same_typenum_as(dataT_typenum, {sample_weight, centroids_t, centroids_half_l2_norm,
    new_centroids_t_private_copies, cluster_sizes_private_copies}))
  {
    throw py::value_error("Array arguments have different elemental data types");
  }

  const auto &api = dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;

  if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    comp_ev = lloyd_single_step<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters,
      centroids_window_height, n_copies, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroids_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignments_idx.get_data<indT>(),
      new_centroids_t_private_copies.get_data<dataT>(),
      cluster_sizes_private_copies.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    comp_ev = lloyd_single_step<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters,
      centroids_window_height, n_copies, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroids_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignments_idx.get_data<indT>(),
      new_centroids_t_private_copies.get_data<dataT>(),
      cluster_sizes_private_copies.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    comp_ev = lloyd_single_step<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters,
      centroids_window_height, n_copies, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroids_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignments_idx.get_data<indT>(),
      new_centroids_t_private_copies.get_data<dataT>(),
      cluster_sizes_private_copies.get_data<dataT>(),
      depends
    );
  } else if (dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    comp_ev = lloyd_single_step<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q,
      n_samples, n_features, n_clusters,
      centroids_window_height, n_copies, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), centroids_t.get_data<dataT>(),
      centroids_half_l2_norm.get_data<dataT>(), assignments_idx.get_data<indT>(),
      new_centroids_t_private_copies.get_data<dataT>(),
      cluster_sizes_private_copies.get_data<dataT>(),
      depends
    );
  } else {
    throw py::value_error("Unsupported array elemental data types.");
  }

  sycl::event ht_ev = dpctl::utils::keep_args_alive(q, {
    X_t, sample_weight, centroids_t, centroids_half_l2_norm, assignments_idx,
    new_centroids_t_private_copies, cluster_sizes_private_copies
  }, {comp_ev});

  return std::make_pair(ht_ev, comp_ev);
}

size_t py_compute_number_of_private_copies(
  dpctl::tensor::usm_ndarray arr,   // only used to infer data type and queue
  size_t n_samples, size_t n_features, size_t n_clusters,
  double centroids_private_copies_max_cache_occupancy,
  size_t work_group_size
) {
  int typenum = arr.get_typenum();
  sycl::queue q = arr.get_queue();

  const auto &api = dpctl::detail::dpctl_capi::get();

  if (centroids_private_copies_max_cache_occupancy <= 0.0 || centroids_private_copies_max_cache_occupancy >= 1.0) {
    throw py::value_error("Expecting a fraction strictly between 0 and 1");
  }

  if (typenum == api.UAR_FLOAT_) {
    using T = float;

    size_t n_copies = compute_number_of_private_copies<T, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, work_group_size
    );

    return n_copies;
  }
  else if (typenum == api.UAR_DOUBLE_) {
    using T = double;

    size_t n_copies = compute_number_of_private_copies<T, preferred_work_group_size_multiple, centroids_window_width_multiplier>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, work_group_size
    );

    return n_copies;
  }
  else {
    throw py::value_error("Unsupported elemental data type");
  }
}

std::pair<size_t, py::array>
py_kmeans_lloyd_driver(
  dpctl::tensor::usm_ndarray X_t,
  dpctl::tensor::usm_ndarray sample_weight,
  dpctl::tensor::usm_ndarray init_centroids_t,
  dpctl::tensor::usm_ndarray assignment_id,
  dpctl::tensor::usm_ndarray res_centroids_t,
  double tol,
  bool verbose,
  size_t max_iter,
  size_t centroids_window_height, 
  size_t work_group_size,
  double centroids_private_copies_max_cache_occupancy,
  sycl::queue q,
  const std::vector<sycl::event> &depends = {}
) {

  if (!is_2d(X_t) || !is_1d(sample_weight) || !is_2d(init_centroids_t) || !is_2d(res_centroids_t) || !is_1d(assignment_id)) {
    throw py::value_error("Unsupported array dimensionalities");
  }

  if (!all_c_contiguous({X_t, sample_weight, init_centroids_t, assignment_id, res_centroids_t})) {
    throw py::value_error("All input arrays must be C-contiguous");
  }

  if (!queues_are_compatible(q, {
    X_t, sample_weight, init_centroids_t,
    assignment_id, res_centroids_t
  })) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  py::ssize_t n_features = X_t.get_shape(0);
  py::ssize_t n_samples = X_t.get_shape(1);
  py::ssize_t n_clusters = init_centroids_t.get_shape(1);

  if ( n_features != init_centroids_t.get_shape(0) || n_features != res_centroids_t.get_shape(0) || 
       n_clusters != res_centroids_t.get_shape(1) || n_samples != sample_weight.get_shape(0) ||
       n_samples != assignment_id.get_shape(0)
  ) {
    throw py::value_error("Array dimensions are not consistent");
  }

  int dataT_typenum = X_t.get_typenum();
  int indT_typenum = assignment_id.get_typenum();

  if (!same_typenum_as(dataT_typenum, {sample_weight, init_centroids_t, res_centroids_t})) {
    throw py::value_error("Sample coordinates, weights and centroids must have the same elemental data types");
  }

  if (centroids_private_copies_max_cache_occupancy <= 0.0 || centroids_private_copies_max_cache_occupancy >= 1.0) {
    throw py::value_error("Fraction `centroids_private_copies_max_cache_occupancy` is out of bounds");
  } 

  if (tol < 0.0) {
    throw py::value_error("Tolerance must be non-negative");
  }


  const auto &api = dpctl::detail::dpctl_capi::get();
  auto py_print_fn = [](const std::stringstream &ss) -> void { py::print( ss.str() ); };

  size_t n_iters_;
  py::array py_total_inertia;

  if( dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT32_) {
    using dataT = float;
    using indT = std::int32_t;

    auto tmp = py::array_t<dataT>(1);
    dataT *total_inertia_ptr = tmp.mutable_data(0);
    py_total_inertia = py::cast<py::array>(tmp);

    n_iters_ =  driver_lloyd<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier, decltype(py_print_fn)>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), init_centroids_t.get_data<dataT>(), 
      max_iter, verbose, static_cast<dataT>(tol), 
      assignment_id.get_data<indT>(), res_centroids_t.get_data<dataT>(), *total_inertia_ptr, py_print_fn
    );
  } else if( dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT32_) {
    using dataT = double;
    using indT = std::int32_t;

    auto tmp = py::array_t<dataT>(1);
    dataT *total_inertia_ptr = tmp.mutable_data(0);
    py_total_inertia = py::cast<py::array>(tmp);

    n_iters_ =  driver_lloyd<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier, decltype(py_print_fn)>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), init_centroids_t.get_data<dataT>(), 
      max_iter, verbose, static_cast<dataT>(tol), 
      assignment_id.get_data<indT>(), res_centroids_t.get_data<dataT>(), *total_inertia_ptr, py_print_fn
    );
  } else if( dataT_typenum == api.UAR_FLOAT_ && indT_typenum == api.UAR_INT64_) {
    using dataT = float;
    using indT = std::int64_t;

    auto tmp = py::array_t<dataT>(1);
    dataT *total_inertia_ptr = tmp.mutable_data(0);
    py_total_inertia = py::cast<py::array>(tmp);

    n_iters_ =  driver_lloyd<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier, decltype(py_print_fn)>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), init_centroids_t.get_data<dataT>(), 
      max_iter, verbose, static_cast<dataT>(tol), 
      assignment_id.get_data<indT>(), res_centroids_t.get_data<dataT>(), *total_inertia_ptr, py_print_fn
    );
  } else if( dataT_typenum == api.UAR_DOUBLE_ && indT_typenum == api.UAR_INT64_) {
    using dataT = double;
    using indT = std::int64_t;

    auto tmp = py::array_t<dataT>(1);
    dataT *total_inertia_ptr = tmp.mutable_data(0);
    py_total_inertia = py::cast<py::array>(tmp);

    n_iters_ =  driver_lloyd<dataT, indT, preferred_work_group_size_multiple, centroids_window_width_multiplier, decltype(py_print_fn)>(
      q, n_samples, n_features, n_clusters, centroids_private_copies_max_cache_occupancy, centroids_window_height, work_group_size,
      X_t.get_data<dataT>(), sample_weight.get_data<dataT>(), init_centroids_t.get_data<dataT>(), 
      max_iter, verbose, static_cast<dataT>(tol), 
      assignment_id.get_data<indT>(), res_centroids_t.get_data<dataT>(), *total_inertia_ptr, py_print_fn
    );
  } else {
    throw py::value_error("Unsupport elemental data type");
  }

  return std::make_pair(n_iters_, py_total_inertia);
}

PYBIND11_MODULE(_kmeans_dpcpp, m) {
  m.def(
    "broadcast_divide", &py_broadcast_divide,
          "broadcast_divide(divident=src, divisor=dst, sycl_queue=q, depends=[]) evaluates "
    "`src /= dst` for matrix src and vector dst",
          py::arg("divident"), py::arg("divisor"),
          py::arg("sycl_queue"), py::arg("depends")=py::list()
  );
  m.def(
    "half_l2_norm_squared", &py_half_l2_norm_squared,
          "half_l2_norm_squared(centroids=X, centroids_half_l2_norm_squared=y, sycl_queue=q, depends=[]) "
    "computes row-wise half of norm squared of X and places it in y",
          py::arg("centroids"), py::arg("centroids_half_l2_norm_squared"),
          py::arg("sycl_queue"), py::arg("depends") = py::list()
  );

  m.def(
    "reduce_centroids_data", &py_reduce_centroids_data,
    "reduce_centroids_data(cluster_sizes_private_copies, centroids_t_private_copies, out_cluster_sizes, "
    " out_centroids_t, out_empty_clusters_list, out_n_empty_clusters, sycl_queue=q, depends=[])",
    py::arg("cluster_sizes_private_copies"),  // IN (n_copies, n_clusters)                dataT
    py::arg("centroids_t_private_copies"),    // IN (n_copies, n_features, n_clusters,)   dataT
    py::arg("out_cluster_sizes"),             // OUT (n_clusters,)                        dataT
    py::arg("out_centroids_t"),               // OUT (n_features, n_clusters,)            dataT
    py::arg("out_empty_clusters_list"),       // OUT (n_clusters,)                        indT
    py::arg("out_n_empty_clusters"),          // OUT (1,)                                 indT
    py::arg("work_group_size"),
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def("compute_threshold", &py_compute_threshold,
    "compute_threshold(data, topk, threshold, sycl_queue=q, depends=[]) finds "
    "topk-th largest element in data and puts in threshold",
    py::arg("data"), py::arg("topk"), py::arg("threshold"),
    py::arg("sycl_queue"), py::arg("depends") = py::list()
  );

  m.def(
    "select_samples_far_from_centroid", &py_select_samples_far_from_centroid,
    "select_samples_far_from_centroid(n_selected, distance_to_centroid, threshold, selected_samples_idx, n_selected_gt_threshold, n_selected_eq_threshold, work_group_size, sycl_queue=q, depends=[]) "
    " populates `selected_samples_idx` with ids of observations whose distance to nearest centroid is greater than `threshold`. The last element of `selected_samples_idx` is "
    " populated with id of the observation whose distance to centroid centroid equals to `threshold`. `n_selected_gt_threshold` and `n_selected_eq_threshold` are temporary scalars.",
    py::arg("n_selected"), py::arg("distance_to_centroid"), py::arg("threshold"),
    py::arg("selected_samples_idx"), py::arg("n_selected_gt_threshold"),
    py::arg("n_selected_eq_threshold"), py::arg("work_group_size"),
    py::arg("sycl_queue"), py::arg("depends") = py::list()
  );

  m.def(
    "relocate_empty_clusters", &py_relocate_empty_clusters,
    "Relocates empty clustsers, modifies centroid_t, cluster_sizes, per_sample_inertia",
    py::arg("n_empty_clusters"),    // int
    py::arg("X_t"),                 // IN (n_features, n_samples,)      dataT
    py::arg("sample_weights"),      // IN (n_samples, )                 dataT
    py::arg("assignment_id"),       // IN (n_samples, )                 indT
    py::arg("empty_clusters_list"), // IN (n_clusters, )                indT
    py::arg("sq_dist_to_nearest_centroid"), // IN (n_samples, )         dataT
    py::arg("centroid_t"),          // INTOUT (n_features, n_clusters,) dataT
    py::arg("cluster_sizes"),       // INOUT  (n_clusters, )            dataT
    py::arg("per_sample_inertia"),  // INOUT  (n_samples,)              dataT
    py::arg("work_group_size"),
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "compute_centroid_shifts_squared", &py_compute_centroid_shifts_squared_kernel,
    "Computes equivalent of `np.sum(np.square(old_centroid_t - new_centroid_t), axis=0)",
    py::arg("centroid_t"),          // IN (n_features, n_clusters, )
    py::arg("new_centroid_t"),      // IN (n_features, n_clusters, )
    py::arg("out_centroid_shifts"), // OUT (n_clusters)
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "compute_centroid_to_sample_distances", &py_compute_distances,
    "Computes distances from centroids to samples. "
    "Inputs: X_t - samples with shape (n_features, n_samples), "
    "centroid_t - centroids with shape (n_features, n_clusters), "
    "and output - euclidean_distances_t with shape (n_clusters, n_samples).",
    py::arg("X_t"),                  // IN (n_features, n_samples)
    py::arg("centroid_t"),           // IN (n_features, n_clusters)
    py::arg("euclidean_distances_t"),// OUT (n_clusters, n_samples)
    py::arg("work_group_size"),
    py::arg("centroids_window_height"),
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "assignment", &py_assignment,
    "Compute assignment of samples to nearest centroids.",
    py::arg("X_t"),                     // IN (n_features, n_samples,)
    py::arg("centroids_t"),             // IN (n_features, n_clusters, )
    py::arg("centroids_half_l2_norm"),  // IN (n_clusters, )
    py::arg("assignment_id"),           // OUT (n_samples,)
    py::arg("centroids_window_height"),
    py::arg("work_group_size"),
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "compute_inertia", &py_compute_inertia,
    "Computes per sample inertia given assignment IDs",
    py::arg("X_t"),           // IN (n_features, n_samples)
    py::arg("sample_weight"), // IN (n_samples)
    py::arg("centroid_t"),    // IN (n_features, n_clusters)
    py::arg("assignment_id"), // IN (n_samples)
    py::arg("out_per_sample_inertia"),
    py::arg("work_group_size"),
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "reduce_vector_blocking", &py_reduce_vector_blocking,
    "Synchronously compute the total value of elements in the vector",
    py::arg("data"), py::arg("sycl_queue"), py::arg("depends") = py::list()
  );

  m.def(
    "fused_lloyd_single_step", &py_fused_lloyd_single_step,
    "Perform single step of Lloyd' algorithm for KMeans problem",
    py::arg("X_t"),                      // IN
    py::arg("sample_weight"),            // IN
    py::arg("centroids_t"),              // IN
    py::arg("centroids_half_l2_norm"),   // IN
    py::arg("assignments_idx"),                 // OUT
    py::arg("new_centroids_t_private_copies"),  // OUT
    py::arg("cluster_sizes_private_copies"),    // OUT
    py::arg("centroids_window_height"),  // size_t
    py::arg("work_group_size"),          // size_t
    py::arg("sycl_queue"),
    py::arg("depends") = py::list()
  );

  m.def(
    "compute_number_of_private_copies",
    &py_compute_number_of_private_copies,
    "Computes optimal number of private copies parametrized by max-cache-ocupancy fraction.",
    py::arg("array"),  // Any array carrying data-type of interest, and allocation queue
    py::arg("n_samples"), py::arg("n_features"), py::arg("n_clusters"),
    py::arg("centroids_private_copies_max_cache_occupancy"), py::arg("work_group_size")
  );

  // returns (ht_ev, comp_ev, n_iters_, total_inertia_, )
  m.def(
    "kmeans_lloyd_driver",
    &py_kmeans_lloyd_driver,
    "Implement Lloyd's refinement algorithm. "
    "Returns 2-tuple, number of iterations performed and 0d numpy array with total_inertia "
    "of the returned configuration. "
    ""
    "Array init_centroid_t is overwritten.",
    py::arg("X_t"),             // IN        (n_features, n_samples, )
    py::arg("sample_weight"),   // IN        (n_sample, )
    py::arg("init_centroid_t"), // IN-OUT    (n_features, n_clusters,)
    py::arg("assignments_id"),  // OUT       (n_samples, )
    py::arg("res_centroids_t"), // OUT       (n_features, n_clusters,)
    py::arg("tol"),             // double
    py::arg("verbose"),         // bool
    py::arg("max_iter"),        // size_t
    py::arg("centroids_window_height"),  // size_t
    py::arg("work_group_size"),
    py::arg("centroids_private_copies_max_cache_occupancy"), // double, fraction in (0, 1)
    py::arg("sycl_queue"), 
    py::arg("depends") = py::list()
  );
}
