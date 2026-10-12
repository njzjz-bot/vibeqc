# Fixed orbital-f NaH HF quartet: pre-acquisition review

Status: input/generator preparation only. No PySCF acquisition has run for this
slice, and no E/F, SCF, or gradient reference values have been created. The
existing numerical consumer remains live until genuine independent receipts
have been reviewed and authenticated.

## Exact scientific scope

The sole eventual consumer is
`tests/python/test_ecp_f.py::test_f_complete_hf_energy_and_force`.
The four inputs are RHF/UHF × Cartesian/spherical NaH in Bohr, with the original
Na LANL2DZ/H STO-3G basis, Na LANL2DZ ECP, and exactly one extra Na orbital-f
shell `[3, [0.55, 0.81], [0.17, 0.23]]`. There is no extra d shell, H f shell,
f projector, compact-oracle basis, displacement, alternative initial guess,
SCF polish, or retry. Cartesian/spherical AO dimensions are 19/16.

The RHF case is neutral with populations (1,1). The UHF case has charge=spin=1
and populations (1,0). The exact original call remains
`(scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)`.

Important upstream behavior: in pinned PySCF 2.14.0, the one-electron UHF
factory returns `pyscf.scf.uhf.HF1e`. Its default `init_guess` property still
reads `minao`, but its analytic one-electron solve does not use that guess and
has zero iterative SCF cycles. The generator preserves this factory dispatch,
records its actual class and cycles, and does not replace it with iterative UHF.
The full HF1e alpha/beta coefficients and orbital energies are those returned by
the original one-electron diagonalization. They must not be silently replaced
with a later Fock-canonicalized virtual spectrum.

The resolved basis/ECP inputs are copied from the already authenticated raw-ECP
receipt and extended by the explicit shell above. These are input parameters,
not inferred new scientific outputs. Both new processes independently reload
installed, byte-pinned upstream basis files and must reconstruct them exactly.

## Bounded acquisition plan

Use only the separate `njzjz-bot/vibeqc` branch
`oracle/ecp-f-hf-reference-20261012`, after independent review. A temporary
branch-guarded ubuntu-24.04 CPU workflow has a 10-minute job limit. It installs
only Python 3.12.12, PySCF 2.14.0, NumPy 2.3.5, SciPy 1.17.0, h5py 3.16.0, and
threadpoolctl 3.6.0, and setuptools 83.0.0. All native thread limits are one; CUDA
devices are hidden.
There is no GQC installation, native build, GPU campaign, canonical acquisition
workflow, auth change, or cache deletion. The timeout is a starting bound,
not a measured runtime claim.

For each `rhf-cartesian`, `rhf-spherical`, `uhf-cartesian`, `uhf-spherical`, run
`tools/generate_ecp_f_hf_reference.py --profile PROFILE --replica first --output
oracle-receipts/PROFILE-first.json` and then the same in another fresh interpreter
with `second`. Thus there are eight fresh processes, two per exact input.
`--verify` requires all eight original files and emits `manifest.json` only if
all full arrays agree within absolute 1e-13 and molecule/SCF/provenance identities
agree. It keeps both original receipts, their distinct process IDs and timings,
file byte counts/hashes, and per-array replay differences. Outputs never overwrite
existing files.

The acquisition must retain successful GitHub run/job/checkout/artifact identities
and verify the downloaded archive against the successful run. Published numerical
data will be the exact downloaded bytes, never values copied from GQC or logs.
No new numerical data or `manifest.json` is supplied at this preparation stage.

## Complete records and provenance

Each profile retains 16 complete FP64 arrays/scalars with exact shape, byte count,
canonical little-endian C-order base64, and SHA256:

- Full overlap, hcore, raw ECP and normalized ECP matrices, plus AO normalizations
- Full density, Fock, coefficients, and commutator residual, with two complete
  spin blocks for UHF and one actual restricted block for RHF
- All orbital energies, occupations, and occupied–virtual orbital gradients
- HF and nuclear energies, full analytic gradient (2,3), and full forces (2,3)

The endpoint does not consume a four-index ERI tensor, so none is added.
No scalar norm replaces a matrix or vector. The receipt also retains explicit
resolved molecule/basis/ECP data, core/effective charges, shell/AO locations,
atom AO slices, structured and printed AO labels, convergence, actual solver
class, cycles, and resolved SCF settings. No orbital phase/rotation adjustment
is applied between processes.

The generator validates full density reconstruction, coefficient metric
orthonormality, electron populations, energy reconstruction from density/Fock,
commutator reconstruction, exact gradient/force sign, full ECP normalization,
and the original net-force bound. It checks finite complete array transport,
exact inventory, input pins, explicit empty regular PySCF configuration,
distribution-owned imports, installed package/native hashes, pinned upstream
SCF/factory and basis files, actual fork checkout/run, and single-thread native
providers. Imports of the GQC implementation and GPU scientific packages are
forbidden. Acquisition source hashes are historical regeneration gates only;
ordinary tests do not require current production function-body hashes.

`scf_seconds` measures the original solver `.run(...)` call. `gradient_seconds`
measures `nuc_grad_method().kernel()`. These exclude package provenance scanning,
molecule preparation and archive/serialization work. Both replicas' observations
are retained separately. The prior 132.26-second RHF/cartesian hosted duration
was the whole GQC+PySCF test, and is not a reference-time or speedup measurement.

## Reader and consumer design for the next reviewed stage

Add `tests/python/_ecp_f_hf_reference.py`, a bounded stdlib-only reader analogous
to the delivered Li/ECP readers. It will:

1. Read the exact inputs, manifest and eight original acquisition files with
   byte limits, duplicate-key/nonfinite rejection, safe allowlisted filenames,
   complete inventory, and independent file/array hashes
2. Validate historical acquisition provenance from the retained manifest, without
   comparing its original consumer/generator hashes to later source refactors
3. Validate both originals, their distinct processes, both full arrays and replay
   metadata, resolved AO/basis/ECP identity, solver dispatch, and full force sign
4. Return exact basis/ECP/atom inputs and full first-replica arrays for only the
   requested spin/representation; unknown selectors fail rather than falling
   back to live PySCF

The consumer will use the existing `_record_from_resolved_basis(..., frozen=True)`
converter to build the same GQC input, then run the original Calculator on every
invoked device. Only its local fixture/oracle creation is replaced. Preserve
`result.converged`, energy atol=2e-8/rtol=0, every force component at
atol=2e-6/rtol=0, and net force atol=2e-7. Keep parameter ordering, node IDs,
`require_device`, all event/device deselections, and resource/backend tests.
No CI variants are restored as part of this migration.

Keep shared `fixture`, raw-f all-center finite differences (2e-4 and 7e-5),
detached-center derivatives, moved-coordinate prepared replay and CUDA directional
finite differences live and unchanged. No global fake PySCF module or molecule.

After authentic acquisition, add reader negative controls (missing/extra/truncated
arrays, wrong shape/dtype/hash, swapped profiles, wrong original/source provenance,
invalid processes/replay, tampered full force component or sign, corrupted basis/
AO identity) and run all four frozen CPU consumer profiles with PySCF imports and
import-based skips blocked. Preserve the unchanged dynamic tests by one-off diff
inspection. Check the final delta against then-current master/live PR overlap
before publication and merge.

## Preparation evidence

Live master `44d8c91d80c5c341cd9ce98e86b52b512ce1d444` was checked on 2026-10-12.
The local preparation worktree base is `c7bf888361f57685b6c21cf7922c2f95df6c7bf8`;
the two ECP consumer files and two reused acquisition helpers were independently
fetched at that live master and verified byte-identical to this checkout.
The earlier live open-PR overlap audit found no ECP-f/ECP-heavy candidate overlap.
No existing source/consumer or permanent workflow has been changed here.

The preparation suite currently passes 63 transport/input/provenance-boundary
checks in a PySCF-free environment. These are generator preflight tests, not
scientific endpoint qualification. Synthetic arrays appear only inside isolated
codec rejection tests and are never written as reference fixtures.


## Provenance review repair

The verifier now recomputes every complete package identity using the installed
wheel RECORD plus all installed Python/native-file bytes and counts. Matching
version strings alone cannot admit invented package hashes or inventories.
It also reobserves loaded CPU BLAS/OpenMP provider metadata, thread count and
native bytes, rather than checking only the syntax of retained library names and
hashes. Only known CPU threadpool APIs running at exactly one thread are accepted.
Replay verification initializes the same pinned package/provider imports under a
fresh empty PySCF config, without constructing a molecule or executing SCF or a
gradient. The chemistry, factory call and all scientific outputs are unchanged.

Setuptools is an unconditional dependency in the official PySCF 2.14.0
[pyproject.toml](https://github.com/pyscf/pyscf/blob/v2.14.0/pyproject.toml).
It is now pinned and covered by the same complete package identity checks.
The official [PyPI setuptools 83.0.0 release](https://pypi.org/project/setuptools/83.0.0/)
provides the universal Python-3 wheel and requires Python >=3.10, compatible with
the pinned Python 3.12.12. Its published wheel SHA256 is
`29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3`.
This is a deliberately selected verified release, not a latest-version claim.

Positive controlled-evidence checks and in-memory mutations cover full package
identities, invented/CUDA/provider metadata, actual multithread or non-CPU
observations, and native-file byte changes. A separate local observation exercised
real NumPy OpenBLAS at one thread, without importing PySCF or running any solve.
None of these controls creates an acquired scientific fixture.
