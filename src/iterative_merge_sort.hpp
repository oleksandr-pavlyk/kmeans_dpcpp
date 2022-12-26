#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <vector>
#include "quotients_utils.hpp"

namespace {

size_t greatest_power_of_two_no_greater_than(size_t n)
{
	if (0uL == (n & (n - 1)))
	{
		return n;
	}

	constexpr size_t largest_power_of_two =
		(size_t(1) << (8 * sizeof(size_t) - 1));
	if (n > largest_power_of_two)
	{
		return largest_power_of_two;
	}

	size_t k = 1;
	while (k < n)
	{
		k <<= 1;
	}
	return k >> 1;
}

template <typename Acc, typename Value, typename Compare>
std::size_t lower_bound_impl(Acc acc, std::size_t first, std::size_t last,
                             const Value &value, Compare comp) {
  std::size_t n = last - first;
  std::size_t cur = n;
  std::size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (comp(acc[it], value)) {
      n -= cur + 1, first = ++it;
    } else
      n = cur;
  }
  return first;
}

template <typename Acc, typename Value, typename Compare>
std::size_t upper_bound_impl(Acc acc, const std::size_t first,
                             const std::size_t last, const Value &value,
                             Compare comp) {
  return lower_bound_impl(acc, first, last, value,
                          [comp](auto x, auto y) { return !comp(y, x); });
}

template <typename Ptr, typename Compare>
void merge_impl(const std::size_t offset, Ptr in_acc, Ptr out_acc,
                const std::size_t start_1, const std::size_t end_1,
                const std::size_t end_2, const std::size_t start_out,
                Compare comp, const std::size_t chunk) {
  const std::size_t start_2 = end_1;
  // Borders of the sequences to merge within this call
  const std::size_t local_start_1 = sycl::min(offset + start_1, end_1);
  const std::size_t local_end_1 = sycl::min(local_start_1 + chunk, end_1);
  const std::size_t local_start_2 = sycl::min(offset + start_2, end_2);
  const std::size_t local_end_2 = sycl::min(local_start_2 + chunk, end_2);

  const std::size_t local_size_1 = local_end_1 - local_start_1;
  const std::size_t local_size_2 = local_end_2 - local_start_2;

  const auto r_item_1 = in_acc[end_1 - 1];
  const auto l_item_2 = in_acc[start_2];

  // Copy if the sequences are sorted with respect to each other or merge otherwise
  if (!comp(l_item_2, r_item_1)) {
    const std::size_t out_shift_1 = start_out + local_start_1 - start_1;
    const std::size_t out_shift_2 =
        start_out + end_1 - start_1 + local_start_2 - start_2;

    for (std::size_t i = 0; i < local_size_1; ++i) {
      out_acc[out_shift_1 + i] = in_acc[local_start_1 + i];
    }
    for (std::size_t i = 0; i < local_size_2; ++i) {
      out_acc[out_shift_2 + i] = in_acc[local_start_2 + i];
    }
  } else if (comp(r_item_1, l_item_2)) {
    const std::size_t out_shift_1 =
        start_out + end_2 - start_2 + local_start_1 - start_1;
    const std::size_t out_shift_2 = start_out + local_start_2 - start_2;
    for (std::size_t i = 0; i < local_size_1; ++i) {
      out_acc[out_shift_1 + i] = in_acc[local_start_1 + i];
    }
    for (std::size_t i = 0; i < local_size_2; ++i) {
      out_acc[out_shift_2 + i] = in_acc[local_start_2 + i];
    }
  }
  // Perform merging
  else {

    // Process 1st sequence
    if (local_start_1 < local_end_1) {
      // Reduce the range for searching within the 2nd sequence and handle bound
      // items find left border in 2nd sequence
      const auto local_l_item_1 = in_acc[local_start_1];
      std::size_t l_search_bound_2 =
          lower_bound_impl(in_acc, start_2, end_2, local_l_item_1, comp);
      const std::size_t l_shift_1 = local_start_1 - start_1;
      const std::size_t l_shift_2 = l_search_bound_2 - start_2;

      out_acc[start_out + l_shift_1 + l_shift_2] = local_l_item_1;

      std::size_t r_search_bound_2{};
      // find right border in 2nd sequence
      if (local_size_1 > 1) {
        const auto local_r_item_1 = in_acc[local_end_1 - 1];
        r_search_bound_2 = lower_bound_impl(in_acc, l_search_bound_2, end_2,
                                            local_r_item_1, comp);
        const auto r_shift_1 = local_end_1 - 1 - start_1;
        const auto r_shift_2 = r_search_bound_2 - start_2;

        out_acc[start_out + r_shift_1 + r_shift_2] = local_r_item_1;
      }


      // Handle intermediate items
      if (r_search_bound_2 == l_search_bound_2) {
        const std::size_t shift_2 = l_search_bound_2 - start_2;
        for (std::size_t idx = local_start_1 + 1; idx < local_end_1 - 1;
             ++idx) {
          const auto intermediate_item_1 = in_acc[idx];
          const std::size_t shift_1 = idx - start_1;
          out_acc[start_out + shift_1 + shift_2] = intermediate_item_1;
        }
      } else {
        for (std::size_t idx = local_start_1 + 1; idx < local_end_1 - 1;
             ++idx) {
          const auto intermediate_item_1 = in_acc[idx];
          // we shouldn't seek in whole 2nd sequence. Just for the part where
          // the 1st sequence should be
          l_search_bound_2 =
              lower_bound_impl(in_acc, l_search_bound_2, r_search_bound_2,
                               intermediate_item_1, comp);
          const std::size_t shift_1 = idx - start_1;
          const std::size_t shift_2 = l_search_bound_2 - start_2;

          out_acc[start_out + shift_1 + shift_2] = intermediate_item_1;
        }
      }
    }
    // Process 2nd sequence
    if (local_start_2 < local_end_2) {
      // Reduce the range for searching within the 1st sequence and handle bound
      // items find left border in 1st sequence
      const auto local_l_item_2 = in_acc[local_start_2];
      std::size_t l_search_bound_1 =
          upper_bound_impl(in_acc, start_1, end_1, local_l_item_2, comp);
      const std::size_t l_shift_1 = l_search_bound_1 - start_1;
      const std::size_t l_shift_2 = local_start_2 - start_2;

      out_acc[start_out + l_shift_1 + l_shift_2] = local_l_item_2;

      std::size_t r_search_bound_1{};
      // find right border in 1st sequence
      if (local_size_2 > 1) {
        const auto local_r_item_2 = in_acc[local_end_2 - 1];
        r_search_bound_1 = upper_bound_impl(in_acc, l_search_bound_1, end_1,
                                            local_r_item_2, comp);
        const std::size_t r_shift_1 = r_search_bound_1 - start_1;
        const std::size_t r_shift_2 = local_end_2 - 1 - start_2;

        out_acc[start_out + r_shift_1 + r_shift_2] = local_r_item_2;
      }

      // Handle intermediate items
      if (l_search_bound_1 == r_search_bound_1) {
        const std::size_t shift_1 = l_search_bound_1 - start_1;
        for (auto idx = local_start_2 + 1; idx < local_end_2 - 1; ++idx) {
          const auto intermediate_item_2 = in_acc[idx];
          const std::size_t shift_2 = idx - start_2;
          out_acc[start_out + shift_1 + shift_2] = intermediate_item_2;
        }
      } else {
        for (auto idx = local_start_2 + 1; idx < local_end_2 - 1; ++idx) {
          const auto intermediate_item_2 = in_acc[idx];
          // we shouldn't seek in whole 1st sequence. Just for the part where
          // the 2nd sequence should be
          l_search_bound_1 =
              upper_bound_impl(in_acc, l_search_bound_1, r_search_bound_1,
                               intermediate_item_2, comp);
          const std::size_t shift_1 = l_search_bound_1 - start_1;
          const std::size_t shift_2 = idx - start_2;

          out_acc[start_out + shift_1 + shift_2] = intermediate_item_2;
        }
      }
    }
  }
}

} // end of anonymous namespace

template <typename T>
sycl::event
iterative_merge_sort(
    sycl::queue &q,
    T const *input,
    T *output,
    size_t n,
    const std::vector<sycl::event> &depends = {})
{
    auto dev = q.get_device();
    size_t lws = std::min(
        dev.get_info<sycl::info::device::max_work_group_size>(),
        dev.get_info<sycl::info::device::local_mem_size>() / (2 * sizeof(T))
    ) / 2;
    constexpr size_t segment_size = 4;

    lws = greatest_power_of_two_no_greater_than(lws / segment_size) * segment_size;
    size_t n_groups = quotient_ceil(n, lws);

    sycl::event base_sort_ev =
        q.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(depends);

                sycl::range<1> global_range{n_groups * lws};
                sycl::range<1> local_range{lws};

                using Comparer = std::less<T>;
                using Sorter = 
                    sycl::ext::oneapi::experimental::default_sorter<Comparer>;
		
                // calculate required local memory size
                // MUST pass range object, not an integer.
                size_t temp_memory_size =
                    Sorter::template memory_required<T>(
                        sycl::memory_scope::work_group, local_range);

                sycl::local_accessor<std::byte, 1>
                    scratch({temp_memory_size}, cgh);

                if (n % lws == 0)
                {
                    cgh.parallel_for(
                        sycl::nd_range<1>(global_range, local_range),
                        [=](sycl::nd_item<1> it)
                        {
                            auto sorter_op = Sorter(
                                sycl::span<std::byte>{
                                    scratch.get_pointer(),
                                    temp_memory_size});
                            output[it.get_global_id()] =
                                sycl::ext::oneapi::experimental::sort_over_group(
                                    it.get_group(),
                                    input[it.get_global_id()],
                                    sorter_op);
                        });
                }
                else
                {
                    cgh.parallel_for(
                        sycl::nd_range<1>(global_range, local_range),
                        [=](sycl::nd_item<1> it)
                        {
                            auto sorter_op = Sorter(
                                sycl::span<std::byte>{
                                    scratch.get_pointer(),
                                    temp_memory_size});
                            const T inp_v = (it.get_global_id(0) < n) ? input[it.get_global_id()] : std::numeric_limits<T>::max();
                            T out_v =
                                sycl::ext::oneapi::experimental::sort_over_group(
                                    it.get_group(),
                                    inp_v,
                                    sorter_op);
                            if (it.get_global_id(0) < n)
                            {
                                output[it.get_global_id()] = out_v;
                            }
                        });
                }
            });

    if (n_groups == 1)
    {
        return base_sort_ev;
    }

    T *src = output;
    T *allocated_mem = sycl::malloc_device<T>(n, q);

    T *dst = allocated_mem;

    bool needs_copy = true;

    const size_t chunk = segment_size;
    size_t sorted_size = lws / segment_size;

    sycl::event dep_ev = base_sort_ev;
    while (sorted_size * chunk < n)
    {
        sycl::event local_dep = dep_ev;

        sycl::event merge_ev = 
            q.submit(
                [&](sycl::handler &cgh) {
                    cgh.depends_on(local_dep);

                    cgh.parallel_for(
                        {quotient_ceil(n, segment_size)}, 
                        [=](sycl::id<1> wid) {
                            auto idx = wid[0];

                            const std::size_t idx_mult = (idx / sorted_size) * sorted_size;
                            const std::size_t idx_rem = (idx - idx_mult);
                            const std::size_t start_1 = sycl::min(2 * idx_mult * chunk, n);
                            const std::size_t end_1 = sycl::min(start_1 + sorted_size * chunk, n);
                            const std::size_t end_2 = sycl::min(end_1 + sorted_size * chunk, n);
                            const std::size_t offset = chunk * idx_rem;

                            T* local_src = const_cast<T *>(src);
                            T* local_dst = const_cast<T *>(dst);
                            std::less<T> comp = std::less<T>();

                            merge_impl<T*, decltype(comp)>(
                                offset, local_src, local_dst, 
                                start_1, end_1, end_2, start_1, 
                                comp, chunk    
                            );
                        });
                });

        sorted_size *= 2;
        dep_ev = merge_ev;

        if (sorted_size * chunk < n)
        {
            std::swap(src, dst);
            needs_copy = !needs_copy;
        }
    }

    if (needs_copy)
    {
        sycl::event copy_ev = q.copy<T>(dst, output, n, dep_ev);
        dep_ev = copy_ev;
    }

    if (allocated_mem)
    {
        sycl::context ctx = q.get_context();
        q.submit(
            [&](sycl::handler &cgh)
            {   
                cgh.depends_on(dep_ev);
                cgh.host_task(
                    [=]()
                    { sycl::free(allocated_mem, ctx); });
            });
    }

    return dep_ev;
}
