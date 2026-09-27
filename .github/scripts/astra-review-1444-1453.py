"""One-task review harness; retained on the fork scratch branch only."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path.cwd()
OUT = ROOT / '_review/astra-1444-1453'
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED = {
    1444: '155e0b0ac50b5279a35846379f8fa7fb5fb610c3',
    1448: '1434916287df7c2ce1ceb42c4adf0d48fe0529c0',
    1452: '0f03dd5993bd7dcc0e58b20afdf6cae57df815df',
    1453: '344b6fa480b078c1fcd7a7298dd7c9fca716bd97',
}

def run(*args, cwd=ROOT, check=True, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError(f'{args}: {result.returncode}\n{result.stdout}')
    return result

def git(*args, **kw):
    return run('git', *args, **kw)

def show(ref, path):
    return git('show', f'{ref}:{path}').stdout

def worktree(number):
    path = Path('/tmp') / f'astra-pr-{number}'
    git('worktree', 'add', '--detach', str(path), EXPECTED[number])
    return path

master = git('rev-parse', 'upstream/master').stdout.strip()
summary = {'master': master, 'prs': {}}
for number, head in EXPECTED.items():
    actual = git('rev-parse', f'upstream/pr-{number}').stdout.strip()
    if actual != head:
        raise RuntimeError(f'PR {number} moved: {actual}')
    base = git('merge-base', head, master).stdout.strip()
    merge = git('merge-tree', '--write-tree', '--messages', head, master, check=False)
    if merge.returncode not in (0, 1):
        raise RuntimeError(merge.stdout)
    tree = merge.stdout.splitlines()[0]
    changed = git('diff', '--name-only', base, head).stdout.splitlines()
    upstream_changed = git('diff', '--name-only', base, master).stdout.splitlines()
    overlap = sorted(set(changed) & set(upstream_changed))
    reports = [f'head {head}\nmaster {master}\n', merge.stdout]
    for path in overlap:
        reports.append(f'\n=== HEAD VERSUS MASTER: {path} ===\n')
        reports.append(git('diff', head, master, '--', path).stdout)
        content = git('show', f'{tree}:{path}', check=False).stdout
        blocks = list(re.finditer(r'(?m)^<<<<<<< .*?^>>>>>>> .*$', content, re.S))
        for block in blocks:
            reports.append(f'\n=== CONFLICT: {path} ===\n{block.group(0)}\n')
    (OUT / f'{number}-compact.txt').write_text(''.join(reports))
    summary['prs'][str(number)] = {'head': head, 'base': base, 'clean': merge.returncode == 0, 'overlap': overlap}

# Reproduce the independently identified assessment-to-source provenance defect.
xwork = worktree(1448)
env = dict(os.environ, PYTHONPATH=str(xwork / 'python'))
probe = '''import json, runpy
ns = runpy.run_path('tests/python/test_dft_xc_program_region.py')
p = ns['_program'](spins=2)
h = ns['_assessment'](ns['HOST_UNFUSED'], spins=1)
d = ns['_assessment'](ns['DEVICE_FUSED'], spins=1)
s = ns['select_native_ks_xc_region_program'](p, host_unfused=h, device_fused=d, device_xc_identity='mismatched-rks-device', endpoint_seconds={'host_unfused': 1.0, 'device_fused': 0.1})
print(json.dumps({'source_spins': 2, 'assessment_spins': 1, 'selected': s.candidate.name, 'source_bytes': next(b.bytes for b in p.buffers if b.name == 'density_host')}))
assert s.candidate.name == 'device_fused', 'The expected old defect did not reproduce'
'''
probe_result = run(sys.executable, '-c', probe, cwd=xwork, env=env, check=False)
(OUT / '1448-provenance-reproduction.txt').write_text(probe_result.stdout)
summary['provenance_probe_exit'] = probe_result.returncode

# 1453 is a mechanical owner migration. Preserve every master algorithm change,
# then apply only the reviewed type/forward-namespace renames to overlap files.
hwork = worktree(1453)
merged = git('merge', '--no-ff', '--no-commit', master, cwd=hwork, check=False)
if merged.returncode not in (0, 1):
    raise RuntimeError(merged.stdout)
conflicts = git('diff', '--name-only', '--diff-filter=U', cwd=hwork).stdout.splitlines()
if set(conflicts) - {'src/cc/rccsdt_force.cpp', 'src/cc/rccsdt_force.hpp', 'src/methods/rccsd_method.cpp'}:
    raise RuntimeError(f'Unexpected conflicts: {conflicts}')
for path in summary['prs']['1453']['overlap']:
    if path not in {'src/cc/rccsdt_force.cpp', 'src/cc/rccsdt_force.hpp', 'src/methods/rccsd_method.cpp'}:
        raise RuntimeError(f'Unexpected overlapping owner migration: {path}')
    original = show(master, path)
    repaired = original.replace('scf::PhysicalReference', 'hf::PhysicalReference')
    if path.endswith('.hpp'):
        repaired = repaired.replace('namespace vibeqc::scf {\nstruct PhysicalReference;', 'namespace vibeqc::hf {\nstruct PhysicalReference;')
    (hwork / path).write_text(repaired)
    git('add', path, cwd=hwork)
git('diff', '--cached', '--check', cwd=hwork)
if git('diff', '--name-only', '--diff-filter=U', cwd=hwork).stdout.strip():
    raise RuntimeError('Unresolved conflict')
base = summary['prs']['1453']['base']
allowed = set(git('diff', '--name-only', base, EXPECTED[1453]).stdout.splitlines())
actual_paths = set(git('diff', '--cached', '--name-only', master, cwd=hwork).stdout.splitlines())
if actual_paths - allowed:
    raise RuntimeError(f'Unrelated integration paths: {actual_paths - allowed}')
(OUT / '1453-candidate-vs-master.diff').write_text(git('diff', '--cached', master, cwd=hwork).stdout)
env = dict(os.environ, PYTHONPATH=str(hwork / 'python'))
tests = run(sys.executable, '-m', 'pytest', '-q', 'tests/python/test_ri_mp2_reverse_memory.py', 'tests/python/test_electronic_structure_boundaries.py', cwd=hwork, env=env, check=False)
(OUT / '1453-tests.txt').write_text(tests.stdout)
summary['prs']['1453']['tests_exit'] = tests.returncode
if tests.returncode == 0:
    git('commit', '-m', 'fix(hf): preserve current RCCSD force paths during reference migration', '-m', 'Merge current master while retaining its RCCSD/RCCSD(T) force split. Only the PhysicalReference owner and forward namespace are changed in overlapping files; scientific and memory-budget logic is preserved.', '-m', 'Agent: ChatGPT\nModel: GPT-6 Astra Pro', cwd=hwork)
    candidate = git('rev-parse', 'HEAD', cwd=hwork).stdout.strip()
    git('push', 'origin', f'{candidate}:refs/heads/review/astra-1453-candidate', cwd=hwork)
    summary['prs']['1453']['candidate'] = candidate
(OUT / 'round2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
