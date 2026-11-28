#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "core/types.hpp"
#include "methods/dft_method_preparation.hpp"
#include "methods/method.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/ks_host_resources.hpp"

namespace generativeqc::api {

/** Recompute the Direct CUDA energy budget from the physical descriptors.
 * A shape-only planner receipt cannot authorize copying native topology. */
class CudaKsHostInventory {
 public:
  explicit CudaKsHostInventory(const generativeqc_method_descriptor& descriptor) {
    auto& plan = plan_;
    auto options = methods::detail::dft_options(descriptor, GENERATIVEQC_BACKEND_CUDA, plan);
    grid_ = methods::detail::ks_grid_options(descriptor, options, plan);
    if ((plan.semilocal_family != dft::SemilocalFamily::Lda &&
         plan.semilocal_family != dft::SemilocalFamily::Pbe) ||
        plan.automatic_program || plan.generated_split_hybrid || plan.nonlocal_correlation ||
        plan.d4_correction || plan.range_exchange || plan.full_range_exchange != 0 ||
        options.nonlocal_grid_spec ||
        options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
        options.precision_mode != GENERATIVEQC_PRECISION_FP64 || options.preliminary_guess ||
        options.incremental_direct_jk ||
        options.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::DeviceFused)
      throw methods::MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "shared host resources require Direct FP64 CUDA LDA/PBE without a seed");
    options_ = std::move(options);
    spins_ = plan.spin_channels;
    first_derivatives_ = plan.semilocal_family == dft::SemilocalFamily::Pbe;
  }

  void append(const core::System& system) {
    using runtime::size_add;
    using runtime::size_mul;
    std::size_t primitives = 0, angular = 0;
    for (const auto& shell : system.shells) {
      primitives = size_add(primitives, shell.primitives.size());
      angular = std::max(angular, std::size_t{shell.angular_momentum});
    }
    if (angular > 2 || !system.ecp_terms.empty() ||
        std::any_of(system.atoms.begin(), system.atoms.end(),
                    [](const auto& atom) { return atom.ecp_core != 0; }))
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CUDA KS host admission requires all-electron through-d shells");
    const auto topology = size_add(size_mul(system.atoms.size(), sizeof(core::Atom)),
                                   size_add(size_mul(system.shells.size(), sizeof(core::Shell)),
                                            size_mul(primitives, sizeof(core::Primitive))));
    templates_ = size_add(templates_, topology);
    coordinates_ = size_add(coordinates_, size_mul(system.atoms.size(), 6 * sizeof(double)));
    binding_ = std::max(binding_, topology);
    fleet_.append(
        {system.atoms.size(), molecule::ao_count(system), molecule::cartesian_ao_count(system),
         system.shells.size(), primitives, spins_, angular, 0, dft::grid_point_count(system, grid_),
         grid_.radial_points, grid_.version == 4 ? 0 : grid_.angular_polar, options_.xc_tile_points,
         options_.diis_history, options_.max_iterations, first_derivatives_, false, 0},
        1, scf::KsHostExecution::cuda_fused);
    // Rebinding retires the old provider before constructing its replacement.
    fleet_.add_retained(
        methods::detail::cuda_ks_direct_energy_fock_host_capacity(system, options_, plan_));
  }

  /** Templates coexist across the fleet; one replacement binding is live at a time. */
  std::array<std::uint64_t, 2> capacity() const {
    const auto inventory = fleet_.inventory();
    return {runtime::size_add(inventory[12], templates_),
            runtime::size_add(inventory[13], runtime::size_add(coordinates_, binding_))};
  }

 private:
  methods::detail::NativeKsExecutionPlan plan_;
  scf::ScfOptions options_;
  dft::GridSpec grid_;
  std::uint64_t spins_{};
  bool first_derivatives_{};
  scf::KsHostFleetResources fleet_;
  std::size_t templates_{}, coordinates_{}, binding_{};
};

}  // namespace generativeqc::api
