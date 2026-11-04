# Compact occupied DF metric workspace

Bridge-owned full-rank fitted occupied response reserves live metric matrices:
one adjoint and, for implicit spectral sources, one root-construction matrix.
Raw, borrowed, truncated and general panels keep four matrices. Shapes retain
overflow checks; allocation matches admission. Public budgets, precision,
quadrature, equations, thresholds and GPU-model conditions are unchanged.
Bounded fallbacks and allocation retries remain.

## Complete endpoint controls

Each device ran baseline/candidate/candidate/baseline for PBE0 and WB97M-V DF96:
one cold and three warm calls per fresh process/owner, one math thread, frozen
engine-local cold density and unchanged stock quadratures. All **24 populations**
pass the unchanged independent README observable, grid-stability and CPU/GPU
spot gates. Warm median changes are within 0.2%; this is regression evidence,
not a speedup claim for the already admitted ordinary response.

| Device | Method | Baseline warm / s | Candidate warm / s |
| --- | --- | ---: | ---: |
| A100 | PBE0 | 4.275076 | 4.273472 |
| A100 | WB97M-V | 25.946725 | 25.906680 |
| RTX 5090 | PBE0 | 6.759666 | 6.771457 |
| RTX 5090 | WB97M-V | 60.686496 | 60.685030 |
| RTX PRO 6000 | PBE0 | 6.365070 | 6.371062 |
| RTX PRO 6000 | WB97M-V | 45.246167 | 45.243865 |

`report.py` prints every cold time, warm repeat, median and dispersion. The
observations retain work counters, SCF/Fock counts, native/scheduler identities,
every independent sample gate and paired error. Maximum paired force difference:
<1.03e-12 Eh/Bohr. This neither requalifies all README methods nor proves every
GPU beats GPU4PySCF. Native/reference WB97M-V long-range fitting still differs.

## Memory-cliff diagnostic

A separate A100 same-instrumented-binary diagnostic compares 2048/256 force
points under the earlier 64-MiB geometry cap. Both one-cold/one-warm arms pass
unchanged independent numerical gates: **component diagnostics, not promoted
defaults or three-repeat rankings**.

At 2048 points the warm free headroom remains exactly 954,531,840 bytes. Omitting
three dead metric matrices saves **313,810,944 bytes**, so the 740,556,800-byte
occupied factors now fit with the live response storage. The prior diagnostic
refused that route and performed 4,489 fitted-panel GEMMs. The new diagnostic
performs zero of those repeated fits:

| Diagnostic | Warm endpoint / s | Occupied DF response / s |
| --- | ---: | ---: |
| Previous 2048-point diagnostic | 75.572 | 52.846 |
| Compact workspace, 2048 points | 23.625708 | 0.920685 |
| Compact workspace, 256 points | 25.948267 | 0.922557 |

Endpoints retain the separate long-range response; the bundle retains original
diagnostic receipts/source identities. Production force tiles and geometry caps
are unchanged. Larger automatic batches still need joint planning and
cross-device qualification.

## Correctness and retained failures

Each of A100, RTX 5090 and RTX PRO 6000 passes **199** rotated-metric/molecular
budget tests plus **34** existing response tests. They cover symmetric, spectral
and rotated explicit roots, condition numbers through 1e10, signed terms,
ragged panels, final-projection reuse, RHF/UHF, cold/warm/moved geometries and
bounded fallbacks. The spectral probe uses independent long-double adjoints;
molecular forces use independent PySCF references. Both Blackwell devices pass
81 compact-adjoint and one tight-budget memcheck tests, with zero errors and
no API-error filtering. This campaign does not claim a new A100 sanitizer pass.

Retained failures: the first constrained allocation lacked live spectral-root
and metadata headroom; the corrected budget remains below the old four-matrix
minimum. The first A100 spectral probe lacked conda CUDA include/library paths;
setting them made the unchanged source pass. Initial host pytest/pre-commit
module lookups lacked those packages; passing checks use the installed test
environment/pre-commit executable. Failed runs remain in ignored raw artifacts,
not relabelled.

Host validation includes 109 response/resource tests, 63 catalog tests and all
684 CI ownership tests across the original run and one corrected timeout-policy
assertion. The runtime-heavy CI shard previously timed out at 99%, with no
reported assertion failures; it receives five additional minutes, retaining
all tests and ten minutes of job cleanup headroom. Source-bound Libxc metadata
changes only source identities, with no new capability promotion/demotion and
historical native evidence still stale. Required pre-commit checks pass.

## Provenance

The source baseline is `742e970b3337a596fd33881c9001879ef6778c72` plus the
recorded metric-workspace patch. Ordinary core-library SHA256:

- A100: `afef597ae83495907f711f9871110529517abf8c2db86e8abe9743abed6b4c90`.
- Both Blackwell devices: `ef02ef64ee0ac90795b61535dca3e64065d96ba51a7e634a74a827e0ba699c0c`.

The ordinary baselines are the preceding molecular VV10 integration, with
original binary/source identities retained. Compilation uses verified sccache
0.16.0 and preserves existing caches. A100 uses finite LSF job 223904 on gpu22;
Blackwell tests use Slurm 1907/3787 and endpoint controls 1908/3788 on n5/n2.
The A100 job reports DONE; both Blackwell endpoint jobs report COMPLETED. Slurm accounting is disabled; the earlier
test jobs' successful client exits and receipts are retained, with no invented
historical scheduler status. Actual device visibility is preserved throughout. Diagnostic sources and the
ordinary A100 core were restored and verified against their original hashes.

Full raw receipts, forces, traces, failures, build logs and local binary identities
remain in ignored `.artifacts/df-metric-workspace-20261031/`. This compact bundle
cannot reconstruct omitted force arrays.

See the [workspace decision](../../../.agents/notes/implemented/performance/2026-10-31-occupied-response-metric-workspace.md)
and [grid batching investigation](../../../.agents/notes/proposed/2026-10-31-stationary-grid-batching.md).
