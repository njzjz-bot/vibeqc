# Decision: Admit Direct CUDA KS energy fleets on the native host ledger

Status: implemented
Date: 2026-11-28

An explicit `HostResources` pool and an energy-only device plan let the prepared
Direct FP64 LDA/PBE batch enforce the host limit before copying native topology
or creating its device providers. The owner sizes every orbital system using
the same native grid prescription and CUDA host inventory as preparation. It
charges retained templates, the replay coordinates and one replacement
binding in addition to the scientific inventory. The planner and native lease
charge the same pageable Direct Fock envelope. The selected-inventory path
does not authorize the optional MD-J source, so it must not use the larger
radial Fock quote that includes that source. A geometry replacement retires the
old provider before constructing its successor. Retained bytes add across live
batches; one workspace lock serializes preparation and replay across contexts.

This is a modeled capacity, not process RSS or an allocator trace. The default
unpooled execution path is unchanged. Fitted Fock, preliminary guesses, hybrid
and other functionals, forces, and response are not admitted by this lease. No
eligible rejection retries unleased preparation.

Qualification against `6b1c07812` on an RTX 5090 (Slurm job `12307`): all five
shared-pool Python tests, 360 adjacent Python regressions, and three native KS
resource CTests passed. The pool tests exercise exact and one-byte-short
admission, zero device allocations on rejection, cold/warm/displaced replay,
and simultaneous retained reservations with a shared workspace.

The formatted candidate was rebuilt with sccache 0.16.0 and requalified in
Slurm job `12318`: five pool tests, 571 adjacent Python tests, and three
native resource CTests passed. A pooled PBE/def2-SVP water calculation
matched independent PySCF at the initial and displaced geometries within
`7.2e-14 Eh`. The adjacent run excluded only the existing
`test_unbuilt_descriptor_capacity_preserves_output_and_validates_abi`:
its unbuilt-profile assumption does not hold in this build with linked CUDA
stationary AOT. Neither the test nor that stationary query changed here.

Submission recheck: job `12329` passed the five pool tests and three native
resource CTests; job `12331` passed the 571 adjacent regressions. The latter
also compared pooled spherical PBE/def2-SVP water against independent PySCF
at the initial and displaced geometries, with a maximum error of `8.6e-14 Eh`.
Set `basis_representation="spherical"` explicitly for this comparison: a named
basis otherwise defaults to Cartesian in the calculator, whereas the PySCF
reference uses spherical functions.

See #2227 (R2).
