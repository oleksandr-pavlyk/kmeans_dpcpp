#include <cstdint>
#include <vector>
#include <utility>
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"
#include "util_kernels.hpp"

namespace py = pybind11;

/*! @brief Evaluates X /= y */
std::pair<sycl::event, sycl::event>
py_broadcast_divide(
  dpctl::tensor::usm_ndarray X,
  dpctl::tensor::usm_ndarray y,
  sycl::queue q,
  const std::vector<sycl::event> &depends={})
{
  int X_ndim = X.get_ndim();
  int y_ndim = y.get_ndim();

  if (X_ndim != 2 || y_ndim != 1 || !X.is_c_contiguous() || !y.is_c_contiguous()) {
    throw py::value_error("Arguments must be a matrix and a vector with C-contiguous layout");
  }

  if (y.get_shape(0) != X.get_shape(1)) {
    throw py::value_error("Array dimensions of arguments are not consistent, X.shape[1] != y.shape[0]");
  }

  if (!dpctl::utils::queues_are_compatible(q, {X.get_queue(), y.get_queue()})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto y_typenum = y.get_typenum();
  if (X.get_typenum() != y_typenum) {
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
  int X_ndim = X.get_ndim();
  int y_ndim = y.get_ndim();

  if (X_ndim != 2 || y_ndim != 1 || !X.is_c_contiguous() || !y.is_c_contiguous()) {
    throw py::value_error("Arguments must be a matrix and a vector with C-contiguous layout");
  }

  if (y.get_shape(0) != X.get_shape(1)) {
    throw py::value_error("Array dimensions of arguments are not consistent, X.shape[1] != y.shape[0]");
  }

  if (!dpctl::utils::queues_are_compatible(q, {X.get_queue(), y.get_queue()})) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto y_typenum = y.get_typenum();
  if (X.get_typenum() != y_typenum) {
    throw py::value_error("Arguments must have the same elemental data types.");
  }

  auto &api = dpctl::detail::dpctl_capi::get();

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
  sycl::queue q,
  const std::vector<sycl::event> &depends={}
) {
  if (cluster_sizes_private_copies.get_ndim() != 2 ||
      centroids_t_private_copies.get_ndim() != 3 ||
      out_cluster_sizes.get_ndim() != 1 ||
      out_centroids_t.get_ndim() != 2 ||
      out_empty_clusters_list.get_ndim() != 1 ||
      out_n_empty_clusters.get_size() != 1)
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

  if (dataT_typenum != centroids_t_private_copies.get_typenum() ||
      dataT_typenum != out_cluster_sizes.get_typenum() ||
      dataT_typenum != out_centroids_t.get_typenum() ||
      indT_typenum != out_empty_clusters_list.get_typenum() )
    {
      throw py::value_error("Array element data types must be consisten");
    }

  if (!cluster_sizes_private_copies.is_c_contiguous() ||
      !centroids_t_private_copies.is_c_contiguous() ||
      !out_cluster_sizes.is_c_contiguous() ||
      !out_centroids_t.is_c_contiguous() ||
      !out_empty_clusters_list.is_c_contiguous() ||
      !out_n_empty_clusters.is_c_contiguous())
  {
    throw py::value_error("All array arguments must be C-contiguous");
  }

  if (! ::dpctl::utils::queues_are_compatible(q, {
						  cluster_sizes_private_copies.get_queue(),
						  centroids_t_private_copies.get_queue(),
						  out_cluster_sizes.get_queue(),
						  out_centroids_t.get_queue(),
						  out_empty_clusters_list.get_queue(),
						  out_n_empty_clusters.get_queue()
      })) {
    throw py::value_error("Execution queue is not compatible with allocation queues");
  }

  auto &api = ::dpctl::detail::dpctl_capi::get();

  sycl::event comp_ev;
  constexpr size_t work_group_size = 256;
  
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
  if (data.get_ndim() !=1 || !data.is_c_contiguous()) {
    throw py::value_error("Argument data must be a C-contiguous vector");
  }

  if (threshold.get_size() != 1) {
    throw py::value_error("Argument threshold must be 1-element array");
  }

  int data_typenum = data.get_typenum();
  if(data_typenum != threshold.get_typenum()) {
    throw py::value_error("Data types of arguments must be the same");
  }

  if (! ::dpctl::utils::queues_are_compatible(q, {data.get_queue(), threshold.get_queue()})) {
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
    using dataT = float;
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
  sycl::queue q,
  const std::vector<sycl::event> &depends = {}
) {

  if (n_selected == 0) { throw py::value_error("Argument `n_selected` must be positive");}

  if (distance_to_centroid.get_ndim() != 1 || selected_samples_idx.get_ndim() != 1 ||
      threshold.get_size() != 1 || n_selected_gt_threshold.get_size() != 1 ||
      n_selected_eq_threshold.get_size() != 1
  ) {
    throw py::value_error("Array dimensionalities are not consistent");
  }

  py::ssize_t n_samples = distance_to_centroid.get_shape(0);
  if ( static_cast<size_t>(n_samples) < n_selected) {
    throw py::value_error("Argument `n_select` is too large");
  }

  if (selected_samples_idx.get_shape(0) < static_cast<py::ssize_t>(n_selected)) {
    throw py::value_error("Vector `selected_samples_idx` must have size of at least `n_selected` elements");
  }

  if (!distance_to_centroid.is_c_contiguous() || !selected_samples_idx.is_c_contiguous()) {
    throw py::value_error("Arrays must be C-contiguous");
  }

  int dataT_typenum = distance_to_centroid.get_typenum();
  if (dataT_typenum != threshold.get_typenum()) {
    throw py::value_error("Input arrays must have the same elemental data type");
  }

  int indT_typenum = selected_samples_idx.get_typenum();

  if (indT_typenum != n_selected_gt_threshold.get_typenum() ||
      indT_typenum != n_selected_eq_threshold.get_typenum()
     ) {
      throw py::value_error("Output array must have the same elemental data type");
  }

  if (!dpctl::utils::queues_are_compatible(q, {
    distance_to_centroid.get_queue(),
    threshold.get_queue(), selected_samples_idx.get_queue(),
    n_selected_gt_threshold.get_queue(),
    n_selected_eq_threshold.get_queue()
  })) { throw py::value_error("Execution queue is not compatible with allocation queues"); }

  auto &api = ::dpctl::detail::dpctl_capi::get();
  size_t work_group_size = 64;

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
    "",
    py::arg("n_selected"), py::arg("distance_to_centroid"), py::arg("threshold"),
    py::arg("selected_samples_idx"), py::arg("n_selected_gt_threshold"), 
    py::arg("n_selected_eq_threshold"), py::arg("sycl_queue"), py::arg("depends") = py::list()
  );
}
