# Decision: reject five-pass bounded force grouping

Status: rejected
Date: 2026-10-04

## Decision

Do not promote or spend further complete-endpoint GPU time on PR #1836's
five-pass grouping. Its fixed-density complete integral-source consumer is
slower both with and without indexed traversal. The production mixed default
is unchanged. This supersedes the
[original proposal](../proposed/2026-10-04-bounded-homogeneous-force-passes.md),
not the separate AO/indexing/Becke endpoint integration.

## Evidence

Frozen candidate `fa1e8f49d924cd7c4bc5edee21240106939edba9`, based on master
`97e55dfc9`, normal-optimizer sm_120 Release build with verified ccache.
Source identity:
`e49832588b19edc85ee9095c4f1110fedcabdc8dabfd6cf824889d0e3cdddc67`.
Final linked library SHA256:
`64eed7b7d3e6f1d01b2a326fe49d38ce84aa45a32a2e1becd48fff34d9ad32a1`.

n1 Slurm5737, one RTX5090 (`CUDA_VISIBLE_DEVICES=4`), finite02:00:00 allocation.
Mixed and homogeneous were alternated in the same process, binary, density and
GPU, three uninstrumented repetitions each. Water96, spherical def2-SVP,
768AO, J=1 and K=-0.125. These are CUDA-event integral-source replay times,
not complete energy/force or SCF endpoints.

| Traversal | Mixed median | Grouped median | Regression |
|---|---:|---:|---:|
| Non-indexed |27.292842s|29.813762s|9.24%|
| Indexed |13.598428s|14.065734s|3.44%|

All55 class ledger entries match exactly in shell, AO and primitive capacities.
The ledger is not executed-root telemetry or a FLOP count. Maximum independent
channel differences are9.131e-12 non-indexed and2.344e-12 indexed, below1e-9.
All four independent CPU-ERI oracle configurations (mixed/grouped times
indexed/off) pass RHF/UKS, pure/mixed sources, opposing spins, Cartesian and
spherical through-f, coincident centers and nonzero screening. The recovered
CI attempt also passes; CI fast-compile is not the measured performance build.

Final linked resource inspection reports fixed4/5/6 stacks of880/1024/1264
bytes instead of the mixed worker's90408 bytes. Registers remain252/255/254,
and the retained high-order pass still reports86920 stack bytes. These static
values do not measure dynamic stack traffic or establish an occupancy win.
Even isolated DPPP and DPDP regress; a small DPPS gain does not rescue the
whole consumer. Do not reopen this solely because a static stack gets smaller.

The failed first allocation, n4 job624, exits before an oracle/kernel run:
NVML580.178 does not match loaded kernel driver580.173.02. The mismatch was not
bypassed or repaired on the shared machine. All qualification above is from
n1; no n4 timings or cross-GPU calibration are used. No n3 GPU jobs.

## Why this did not solve the structural problem

The proposal changes `S + sum(C_order)` to `5S + sum(C_order)` plus four cursor
resets and launches. It does not reduce primitive/AO recurrence work. The
measured regression rejects this execution hypothesis; it does not prove that
all high-order scheduling or compiler optimization is exhausted.

Do not repeat the separately rejected scalar DPPP six-slab/moment cache as a
substitute. A new candidate should actually reduce repeated primitive/root or
component recurrence work, reuse the existing mathematical IR, preserve shared
derivatives with separate J/K outputs, and account for cooperative resource
and screening costs. An isolated class improvement is insufficient.

## Retention and revisit conditions

Raw oracle logs, both all-repeat/channel replay files and manifests are retained
under `.artifacts/qualification-5737/` in the candidate checkout. Build commands,
ccache receipts and final resource report are under `.artifacts/build/`. The
input density SHA256 is
`a9a669914ca2b75be5b6ad74ba168aa4f62cf61b818b274d1a491fae376e9775`.
The unmodified-master control was built on n5 but did not run on a GPU; no
default-control speedup or regression is inferred from that build.

Revisit only with a different work/reuse plan that beats the complete
fixed-density source consumer under identical screening and passes independent
J/K oracle gates. Then require cold, warm, moved and all-repeat full endpoint
qualification before any promotion. Parity remains unproven.
