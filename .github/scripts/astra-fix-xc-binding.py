"""Apply the reviewed XC provenance repair in an isolated PR worktree."""
from pathlib import Path
import re


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def repair(root):
    root = Path(root)
    path = root / 'python/vibeqc_compiler/dft/xc_schedule.py'
    text = path.read_text()
    text = replace_once(text, '\n\n@dataclass(frozen=True)\nclass GridXcScheduleCandidate:', '''

def grid_xc_shape_identity(shape: GridXcCandidateShape) -> str:
    """Identity of the exact candidate-local resource and tiling evidence."""
    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("grid/XC shape identity requires GridXcCandidateShape")
    return canonical_hash(asdict(shape))


def grid_xc_domain_identity(shape: GridXcCandidateShape) -> str:
    """Common workload domain, excluding candidate-local tiling and resources."""
    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("grid/XC domain identity requires GridXcCandidateShape")
    return canonical_hash(
        {
            name: getattr(shape, name)
            for name in ("npoint", "nao", "max_active_ao", "spins", "jet_components")
        }
    )


@dataclass(frozen=True)
class GridXcScheduleCandidate:''')
    text = replace_once(text, '            ("domain_schedule", resolved.name),', '''            ("domain_schedule", resolved.name),
            ("candidate_shape", grid_xc_shape_identity(shape)),
            ("candidate_domain", grid_xc_domain_identity(shape)),''')
    path.write_text(text)

    path = root / 'python/vibeqc_compiler/dft/xc_program.py'
    text = path.read_text()
    text = replace_once(text, 'from .xc_schedule import GridXcCandidateAssessment, GridXcCandidateShape', '''from .xc_schedule import (
    GridXcCandidateAssessment,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    grid_xc_domain_identity,
    grid_xc_shape_identity,
    schedule_profile_key,
)''')
    text = replace_once(text, '    shape: GridXcCandidateShape,\n    *,\n    density_identity: str,', '    shape: GridXcCandidateShape,\n    *,\n    scientific: GridXcScientificIdentity,\n    density_identity: str,')
    text = replace_once(text, '    density_identity = _text(density_identity, "KS density identity")', '''    if not isinstance(scientific, GridXcScientificIdentity):
        raise TypeError("native KS XC source requires scientific identity")
    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("native KS XC source requires candidate shape")
    if (
        shape.spins != (2 if scientific.spin == "polarized" else 1)
        or scientific.observable != "potential"
        or scientific.density_route != "density_matrix"
        or shape.jet_components != len(scientific.jet_outputs)
    ):
        raise ValueError("native KS XC source scientific domain mismatch")
    density_identity = _text(density_identity, "KS density identity")''')
    text = replace_once(text, '    topology = {\n        "npoint": shape.npoint,', '    topology = {\n        "scientific": scientific.identity,\n        "shape": grid_xc_shape_identity(shape),\n        "npoint": shape.npoint,')
    text = replace_once(text, '\n\ndef native_ks_xc_region(program: ProgramIR)', '''

@dataclass(frozen=True, slots=True)
class NativeKsXcSource:
    """DFT-owned source binding used to reconstruct, not relabel, an XC region.

    ProgramIR remains method-neutral. The domain owner carries science and shape
    separately and reconstructs the complete graph before attaching admission
    records; matching just a supplied program digest is not sufficient.
    """

    shape: GridXcCandidateShape
    scientific: GridXcScientificIdentity
    density_identity: str
    host_xc_identity: str
    transfer_identity: str
    device_ordinal: int = 0

    def program(self) -> ProgramIR:
        return native_ks_host_unfused_xc_program(
            self.shape,
            scientific=self.scientific,
            density_identity=self.density_identity,
            host_xc_identity=self.host_xc_identity,
            transfer_identity=self.transfer_identity,
            device_ordinal=self.device_ordinal,
        )


def _validate_source_admissions(
    program: ProgramIR,
    source: NativeKsXcSource,
    host: GridXcCandidateAssessment,
    device: GridXcCandidateAssessment,
) -> None:
    if not isinstance(source, NativeKsXcSource):
        raise TypeError("native KS XC binding requires a typed source")
    if program.identity != source.program().identity:
        raise ValueError("native KS XC source program mismatch")
    scientific = source.scientific
    expected_target = canonical_hash(
        {"backend": "cuda", "architecture": scientific.architecture}
    )
    for assessment in (host, device):
        contract = assessment.schedule_contract
        provenance = dict(contract.provenance)
        if (
            contract.consumer != "dft.grid_xc"
            or assessment.schedule_hash != contract.schedule_hash
            or contract.workload_hash != scientific.identity
            or contract.profile_key != schedule_profile_key(scientific)
            or contract.target_hash != expected_target
            or provenance.get("candidate_domain")
            != grid_xc_domain_identity(source.shape)
        ):
            raise ValueError("grid/XC admission source provenance mismatch")
    if dict(host.schedule_contract.provenance).get(
        "candidate_shape"
    ) != grid_xc_shape_identity(source.shape):
        raise ValueError("host grid/XC admission source shape mismatch")
    if (
        host.schedule_contract.precision_schedule_hash
        != device.schedule_contract.precision_schedule_hash
    ):
        raise ValueError("grid/XC admission source precision mismatch")


def native_ks_xc_region(program: ProgramIR)''')
    old = '    host_unfused: GridXcCandidateAssessment,\n    device_fused: GridXcCandidateAssessment,\n    device_xc_identity: str,'
    assert text.count(old) == 2
    text = text.replace(old, '    source: NativeKsXcSource,\n' + old)
    text = replace_once(text, '    region = native_ks_xc_region(program)\n    host_schedule', '    _validate_source_admissions(program, source, host_unfused, device_fused)\n    region = native_ks_xc_region(program)\n    host_schedule')
    text = replace_once(text, '        program,\n        host_unfused=host_unfused,', '        program,\n        source=source,\n        host_unfused=host_unfused,')
    path.write_text(text)

    path = root / 'tests/python/test_dft_xc_program_region.py'
    text = path.read_text()
    text = replace_once(text, '    bind_native_ks_xc_region_candidates,', '    NativeKsXcSource,\n    bind_native_ks_xc_region_candidates,')
    text = text.replace('    native_ks_host_unfused_xc_program,\n', '')
    text = replace_once(text, '    GridXcCandidateShape,', '    GridXcCandidateShape,\n    GridXcScientificIdentity,')
    text = replace_once(text, '\n\ndef _assessment(', '''

def _scientific(*, spins: int = 2) -> GridXcScientificIdentity:
    return GridXcScientificIdentity(
        architecture="sm_120",
        functional="PBE",
        functional_identity="pbe-science-v1",
        ingredients=("rho", "gradient", "sigma"),
        jet_outputs=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        grid_identity="grid-v1",
        grid_model="grid-model-v1",
        screening_identity=None,
        precision="fp64",
        spin="polarized" if spins == 2 else "unpolarized",
        observable="potential",
        density_route="density_matrix",
        source_identity="xc-source-v1",
    )


def _assessment(''')
    text = replace_once(text, '        functional="PBE",\n    )\n\n\ndef _program', '        functional="PBE",\n        scientific=_scientific(spins=spins),\n    )\n\n\ndef _program')
    begin = text.index('def _program(*, spins: int = 2)')
    end = text.index('\n\ndef test_host_unfused_uks', begin)
    text = text[:begin] + '''def _source(*, spins: int = 2) -> NativeKsXcSource:
    return NativeKsXcSource(
        _shape(spins=spins),
        _scientific(spins=spins),
        density_identity="cuda-ks-density-v1",
        host_xc_identity="host-pbe-v1",
        transfer_identity="cuda-ks-xc-transfer-v1",
    )


def _program(*, spins: int = 2) -> ProgramIR:
    return _source(spins=spins).program()
''' + text[end:]
    text = re.sub(r'(?m)^( +)device_xc_identity=', r'\1source=_source(),\n\1device_xc_identity=', text)
    text += '''

@pytest.mark.parametrize("assessment_spins", [1])
def test_region_rejects_cross_spin_admission(assessment_spins: int) -> None:
    with pytest.raises(ValueError, match="source provenance"):
        select_native_ks_xc_region_program(
            _program(spins=2),
            source=_source(spins=2),
            host_unfused=_assessment(HOST_UNFUSED, spins=assessment_spins),
            device_fused=_assessment(DEVICE_FUSED, spins=assessment_spins),
            device_xc_identity="rks-plan",
            endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.1},
        )


@pytest.mark.parametrize("field,value", [("nao", 8), ("npoint", 64)])
def test_region_rejects_other_shape_admission(field: str, value: int) -> None:
    shape = replace(_shape(), **{field: value})
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule, shape, _limits(), device_xc_available=True,
            observable="potential", functional="PBE", scientific=_scientific(),
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(), source=_source(), host_unfused=assessments[0],
            device_fused=assessments[1], device_xc_identity="other-shape-plan",
        )


@pytest.mark.parametrize("field,value", [
    ("grid_identity", "other-grid"),
    ("source_identity", "other-source"),
    ("functional_identity", "other-functional"),
    ("architecture", "sm_90"),
])
def test_region_rejects_other_scientific_admission(field: str, value: str) -> None:
    science = replace(_scientific(), **{field: value})
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule, _shape(), _limits(), device_xc_available=True,
            observable="potential", functional="PBE", scientific=science,
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(), source=_source(), host_unfused=assessments[0],
            device_fused=assessments[1], device_xc_identity="other-science-plan",
        )


def test_region_rejects_missing_scientific_admission() -> None:
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule, _shape(), _limits(), device_xc_available=True,
            observable="potential", functional="PBE",
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(), source=_source(), host_unfused=assessments[0],
            device_fused=assessments[1], device_xc_identity="unbound-plan",
        )


@pytest.mark.parametrize("field", ["density_identity", "host_xc_identity", "transfer_identity"])
def test_region_rejects_stale_source_binding(field: str) -> None:
    source = replace(_source(), **{field: "changed-identity"})
    with pytest.raises(ValueError, match="source program"):
        bind_native_ks_xc_region_candidates(
            _program(), source=source, host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED), device_xc_identity="device-plan",
        )


def test_region_rejects_reconstructed_graph_with_stale_source() -> None:
    program = _program()
    calls = list(program.calls)
    calls[0] = replace(calls[0], identity="other-transfer")
    modified = replace(program, calls=tuple(calls))
    with pytest.raises(ValueError, match="source program"):
        bind_native_ks_xc_region_candidates(
            modified, source=_source(), host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED), device_xc_identity="device-plan",
        )


def test_region_accepts_candidate_local_device_tiling() -> None:
    device = assess_grid_xc_schedule(
        replace(DEVICE_FUSED, point_tile=8), replace(_shape(), tile_points=8),
        _limits(), device_xc_available=True, observable="potential",
        functional="PBE", scientific=_scientific(),
    )
    selected = select_native_ks_xc_region_program(
        _program(), source=_source(), host_unfused=_assessment(HOST_UNFUSED),
        device_fused=device, device_xc_identity="retiled-plan",
        endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.75},
    )
    assert selected.candidate.name == "device_fused"
'''
    path.write_text(text)
