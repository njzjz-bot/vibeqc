#pragma once

#include <cstddef>

#include "runtime/bounded_workspace.hpp"
#include "scf/df_source_capacity.hpp"

namespace generativeqc::scf {

/** Conservative pageable envelope for one retained Direct CUDA Fock owner,
 * including its optional direct J source and one-electron staging. */
inline std::size_t cuda_ks_direct_fock_host_capacity(const df_source_capacity::Shape& orbital,
                                                     std::size_t provider_bytes,
                                                     std::size_t one_electron_bytes,
                                                     std::size_t ecp_bytes = 0) {
  using runtime::size_add;
  using runtime::size_mul;
  const auto pairs = size_mul(orbital.shells, size_add(orbital.shells, 1)) / 2;
  const auto packing = size_mul(2048, size_add(1, size_add(pairs, size_mul(pairs, pairs))));
  const auto sources = size_add(size_mul(4, provider_bytes), size_mul(4, orbital.system_bytes));
  return size_add(size_add(sources, packing), size_add(one_electron_bytes, ecp_bytes));
}

}  // namespace generativeqc::scf
