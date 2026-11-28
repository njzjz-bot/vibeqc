# Decision: Bind r²SCAN+rVV10 to the native CUDA force owner

Status: implemented
Date: 2026-11-28

Reuse the variational r²SCAN point, prepared full-range J, and resident
nonlocal features through the existing stationary controller. Link only the
corresponding spin profiles in packages that select variational r²SCAN. Check
the nonlocal feature token independently of the optional range provider: this
model has a nonlocal source but no exact-exchange source.

Include the nonlocal variant in the model key so an rVV10 graph cannot borrow a
VV10 binding with coincident scalar parameters. Do not admit rVV10 to the
VV10 exact-component reuse schedule. This changes dispatch, not either pair
equation, and does not broaden budgeted fleet or ECP admission.

The endpoint gate is a complete, independently reconverged energy and full
Richardson force vector on the moved frame. Cover all-electron FP64 RKS/UKS and
Direct/DF-J with shared and independent product grids. Forbid the Python force
controller in the test to establish which production path ran.

See #2227 (R1/R6).
