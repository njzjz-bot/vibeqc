"""Finish the D4 deletion review on an isolated fork-only worktree."""
import hashlib
from pathlib import Path
import re

harness = Path('.github/scripts/astra-review-1444-1453.py').read_text()
exec(compile(harness.split('for number in (1452, 1448, 1444):')[0], 'review_helpers', 'exec'))
number = 1444
try:
    path, conflicts = prepare(number)
    name = 'src/xtb/native/src/backends/cuda/gfn2_d4.cu'
    manifest_name = 'src/xtb/native/CUDA_SOURCE_PROVENANCE.json'
    assert set(conflicts) <= {name, manifest_name}, conflicts
    text = (path / name).read_text()
    block = re.compile(r'^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n', re.M | re.S)
    removed = []
    def resolve(match):
        ours, theirs = match.groups()
        assert not ours.strip(), 'Unexpected nonempty dense-delete conflict side'
        assert 'Gfn2D4DeviceCache' in theirs, 'Unexpected live conflict'
        removed.append(theirs)
        return ours
    repaired, count = block.subn(resolve, text)
    assert count >= 1, 'Expected explicit dead-cache delete/modify conflicts'
    assert not re.search(r'^(<<<<<<<|=======|>>>>>>>)', repaired, re.M)
    (path / name).write_text(repaired)
    git('add', name, cwd=path)
    def tokens(text):
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
        return re.findall(r'\w+|[^\s]', text)
    audit = []
    for source in (name, name + 'h'):
        current = tokens(show(master, source))
        candidate = tokens((path / source).read_text())
        cursor = iter(current)
        assert all(any(item == token for item in cursor) for token in candidate), source
        audit.append(f'{source}: deletion-only token subsequence verified; {len(current)} -> {len(candidate)} tokens')
    # Verify every retained device kernel/helper has unchanged token content.
    function = re.compile(r'(?m)^(?:__global__|__device__)\s+[^;{}]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{')
    def bodies(source):
        result = {}
        for match in function.finditer(source):
            opening = match.end() - 1
            level = 1
            end = opening + 1
            while level and end < len(source):
                level += (source[end] == '{') - (source[end] == '}')
                end += 1
            assert level == 0
            key = match.group(1)
            assert key not in result, f'Unexpected overloaded device function {key}'
            result[key] = tokens(source[match.start():end])
        return result
    old_bodies = bodies(show(master, name))
    new_bodies = bodies(repaired)
    assert new_bodies
    for key, body in new_bodies.items():
        assert key in old_bodies and body == old_bodies[key], f'Retained device function changed: {key}'
    audit.append(f'All {len(new_bodies)} retained device kernels/helpers are token-identical to master')
    audit.append('Deleted device functions: ' + ', '.join(sorted(set(old_bodies) - set(new_bodies))))
    retired = ('Gfn2D4DeviceCache', 'update_gfn2_d4_geometry_cache_cuda', 'evaluate_gfn2_d4_two_body_cuda', 'evaluate_gfn2_d4_scc_potential_cuda', 'evaluate_gfn2_d4_scc_energy_cuda', 'add_gfn2_d4_two_body_gradient_cuda', 'evaluate_gfn2_d4_atm_cuda', 'add_gfn2_d4_atm_gradient_cuda')
    references = git('grep', '-n', '-E', '|'.join(retired), '--', 'src', 'include', cwd=path, check=False)
    assert references.returncode == 1, references.stdout
    audit.append('No retained production references to retired dense API or cache type')
    manifest = json.loads(show(master, manifest_name))
    original = json.loads(show(EXPECTED[number], manifest_name))
    manifest['adaptations']['src/backends/cuda/gfn2_d4.cu'] = original['adaptations']['src/backends/cuda/gfn2_d4.cu']
    for source in ('src/backends/cuda/gfn2_d4.cu', 'src/backends/cuda/gfn2_d4.cuh'):
        manifest['files'][source]['vendored_sha256'] = hashlib.sha256((path / 'src/xtb/native' / source).read_bytes()).hexdigest()
    (path / manifest_name).write_text(json.dumps(manifest, indent=2) + '\n')
    git('add', manifest_name, cwd=path)
    (OUT / '1444-deletion-audit.txt').write_text('\n'.join(audit) + '\n')
    finish(number, path, ['tests/python/test_gfn2_cuda_provenance.py'])
except Exception as exc:
    summary['prs'].setdefault(str(number), {})['error'] = str(exc)
    (OUT / '1444-error.txt').write_text(traceback.format_exc())
(OUT / 'round5-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
