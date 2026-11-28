"""The CUDA energy owner admits and retains native host capacity."""

import os

import numpy as np
import pytest
from generativeqc import Calculator, HostResources

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
HE = [(2, (0.0, 0.0, 0.0))]
H = [(1, (0.0, 0.0, 0.0))]

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU job",
)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_direct_cuda_energy_fleet_host_lease(method: str) -> None:
    """Exact admission survives cold, warm and displaced execution."""
    uks = method.endswith("uks")
    systems = [H2, H if uks else HE]
    inputs = {
        "charges": [1, 0] if uks else [0, 0],
        "multiplicities": [2, 2] if uks else [1, 1],
    }
    calculator = Calculator(method=method, device="cuda", initial_guess=None)
    plan = calculator.estimate_resources(
        systems, properties=("energy",), **inputs
    ).require_feasible()
    assert tuple(plan.requests[0].identity.observables) == ("energy",)
    moved = [[(0.0, 0.0, -0.8), (0.0, 0.0, 0.8)], None]
    with calculator.prepare_batch(systems, resource_plan=plan, **inputs) as plain:
        expected = [
            [
                item.energy
                for item in plain.execute(
                    coords, properties=("energy",), strict=True
                ).items
            ]
            for coords in (None, None, moved)
        ]

    with HostResources(plan.peak_bytes["host"]) as probe:
        with calculator.prepare_batch(
            systems, resource_plan=plan, host_resources=probe, **inputs
        ):
            quoted = probe.snapshot()
            assert quoted.owner_count == 1 and quoted.retained_bytes > 0
            assert quoted.workspace_bytes > 0
        assert probe.snapshot().owner_count == 0
    exact = quoted.retained_bytes + quoted.workspace_bytes

    with HostResources(exact - 1) as short:
        before = short.snapshot()
        with pytest.raises(
            MemoryError, match="shared host capacity exceeded"
        ) as refused:
            calculator.prepare_batch(
                systems, resource_plan=plan, host_resources=short, **inputs
            )
        device = refused.value.resource_diagnostics["preparation"]["device_ledger"]
        assert (
            device["allocations"]
            == device["rejected_allocations"]
            == device["live_bytes"]
            == 0
        )
        assert short.snapshot() == before

    with HostResources(exact) as pool:
        with calculator.prepare_batch(
            systems, resource_plan=plan, host_resources=pool, **inputs
        ) as batch:
            live = pool.snapshot()
            assert live.owner_count == 1
            assert live.retained_bytes + live.workspace_bytes == exact
            with pytest.raises(
                NotImplementedError, match="outside the admitted CUDA KS host"
            ):
                batch.execute(properties=("energy", "forces"))
            for replay, (coords, target) in enumerate(
                zip((None, None, moved), expected, strict=True)
            ):
                outcome = batch.execute(coords, strict=True)
                assert all(item.executed_backend == "cuda" for item in outcome.items)
                if replay == 1:
                    assert all(item.warm_start_used for item in outcome.items)
                np.testing.assert_allclose(
                    [item.energy for item in outcome.items], target, rtol=0, atol=1e-10
                )
                assert pool.snapshot() == live
        cleared = pool.snapshot()
        assert (
            cleared.owner_count
            == cleared.retained_bytes
            == cleared.workspace_bytes
            == 0
        )


def test_direct_cuda_energy_fleets_compose_retained_capacity() -> None:
    """Multiple contexts sum their retained bytes, but share one workspace."""
    water = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 1.43, 1.1)), (1, (0.0, -1.43, 1.1))]
    calculator = Calculator(
        method="pbe-rks", device="cuda", basis="def2-svp", initial_guess=None
    )
    systems = [water]
    plan = calculator.estimate_resources(systems, properties=("energy",))
    plan.require_feasible()
    host_peak = plan.peak_bytes["host"]
    fock_bytes = [
        estimate.bytes
        for candidate in plan.requests[0].candidates
        for estimate in candidate.estimates
        if estimate.name == "all KS Direct Fock host envelopes"
    ]
    assert fock_bytes and len(set(fock_bytes)) == 1
    with (
        HostResources(host_peak) as probe,
        calculator.prepare_batch(systems, resource_plan=plan, host_resources=probe),
    ):
        single = probe.snapshot()
        assert single.retained_bytes > fock_bytes[0] > 0
        assert single.retained_bytes + single.workspace_bytes <= host_peak

    with HostResources(2 * single.retained_bytes + single.workspace_bytes) as pool:
        with calculator.prepare_batch(systems, resource_plan=plan, host_resources=pool):
            with calculator.prepare_batch(
                systems, resource_plan=plan, host_resources=pool
            ):
                both = pool.snapshot()
                assert both.owner_count == 2
                assert both.retained_bytes == 2 * single.retained_bytes
                assert both.workspace_bytes == single.workspace_bytes
                with pytest.raises(MemoryError, match="shared host capacity exceeded"):
                    calculator.prepare_batch(
                        systems, resource_plan=plan, host_resources=pool
                    )
                assert pool.snapshot() == both
            remaining = pool.snapshot()
            assert remaining.owner_count == 1
            assert remaining.retained_bytes == single.retained_bytes
            assert remaining.workspace_bytes == single.workspace_bytes
        assert pool.snapshot().owner_count == 0
