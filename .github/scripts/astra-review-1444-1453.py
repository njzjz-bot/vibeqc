"""One-task review harness; never included in upstream candidate trees."""
import ast
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

ROOT = Path.cwd()
OUT = ROOT / '_review/astra-1444-1453'
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED = {1444:'155e0b0ac50b5279a35846379f8fa7fb5fb610c3', 1448:'1434916287df7c2ce1ceb42c4adf0d48fe0529c0', 1452:'0f03dd5993bd7dcc0e58b20afdf6cae57df815df'}

def run(*args, cwd=ROOT, check=True, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError(f'{args}: {result.returncode}\n{result.stdout}')
    return result

def git(*args, **kw):
    return run('git', *args, **kw)

def show(ref, path):
    return git('show', f'{ref}:{path}').stdout

master = git('rev-parse', 'upstream/master').stdout.strip()
summary = {'master': master, 'prs': {}}

def prepare(number):
    head = EXPECTED[number]
    actual = git('rev-parse', f'upstream/pr-{number}').stdout.strip()
    if actual != head:
        raise RuntimeError(f'PR {number} moved: {actual}')
    path = Path('/tmp') / f'astra-r3-{number}'
    git('worktree', 'add', '--detach', str(path), head)
    merged = git('merge', '--no-ff', '--no-commit', master, cwd=path, check=False)
    if merged.returncode not in (0, 1):
        raise RuntimeError(merged.stdout)
    conflicts = git('diff', '--name-only', '--diff-filter=U', cwd=path).stdout.splitlines()
    summary['prs'][str(number)] = {'head':head, 'conflicts':conflicts}
    report = []
    for name in conflicts:
        content = (path / name).read_text()
        active = False
        report.append(f'\n===== {name} =====\n')
        for line in content.splitlines(keepends=True):
            if line.startswith('<<<<<<< '): active = True
            if active: report.append(line)
            if line.startswith('>>>>>>> '): active = False
    (OUT / f'{number}-conflict-blocks.txt').write_text(''.join(report))
    return path, conflicts

def finish(number, path, tests, extra_paths=()):
    git('add', '-u', cwd=path)
    git('diff', '--cached', '--check', cwd=path)
    assert not git('diff','--name-only','--diff-filter=U',cwd=path).stdout.strip()
    base = git('merge-base',EXPECTED[number],master).stdout.strip()
    allowed = set(git('diff','--name-only',base,EXPECTED[number]).stdout.splitlines()) | set(extra_paths)
    actual = set(git('diff','--cached','--name-only',master,cwd=path).stdout.splitlines())
    assert not actual - allowed, actual - allowed
    (OUT/f'{number}-candidate-vs-master.diff').write_text(git('diff','--cached',master,cwd=path).stdout)
    env = dict(os.environ,PYTHONPATH=str(path/'python'))
    result = run(sys.executable,'-m','pytest','-q',*tests,cwd=path,env=env,check=False)
    (OUT/f'{number}-tests.txt').write_text(result.stdout)
    summary['prs'][str(number)]['tests_exit'] = result.returncode
    if result.returncode:
        return
    git('commit','-m',f'fix(review): repair PR {number} integration and admission contracts','-m','Preserve current master changes, retained scientific assertions and bounded fallbacks. Validation evidence is retained on the separate review branch.','-m','Agent: ChatGPT\nModel: GPT-6 Astra Pro',cwd=path)
    candidate=git('rev-parse','HEAD',cwd=path).stdout.strip()
    git('push','origin',f'{candidate}:refs/heads/review/astra-{number}-candidate',cwd=path)
    summary['prs'][str(number)]['candidate']=candidate

for number in (1452, 1448, 1444):
    try:
        path, conflicts = prepare(number)
        if number == 1452:
            assert set(conflicts) <= {'cmake/VibeQCGeneratedSources.cmake','cmake/VibeQCSourceIdentity.json'}, conflicts
            name='cmake/VibeQCGeneratedSources.cmake'
            own=show(EXPECTED[number],name)
            theirs=show(master,name)
            start=own.index('  if(VIBEQC_ENABLE_STATIONARY_CPU_FORCE_AOT)\n    set(VIBEQC_WB97MV_RSH_CPU_AOT_DIRECTORY')
            end=own.index('  set(VIBEQC_QUADRATURE_HEADER',start)
            assert own[:start]+own[end:] == theirs, 'Unexpected CMake differences outside RSH block'
            (path/name).write_text(own)
            git('add',name,cwd=path)
            name='cmake/VibeQCSourceIdentity.json'
            own=json.loads(show(EXPECTED[number],name))
            theirs=json.loads(show(master,name))
            assert {k:v for k,v in own.items() if k!='files'} == {k:v for k,v in theirs.items() if k!='files'}
            added=[item for item in own['files'] if item not in theirs['files']]
            assert added == ['tools/generate_wb97mv_rsh_cpu_aot.py'], added
            at=theirs['files'].index('tools/generate_stationary_cpu_derivative_aot.py')+1
            theirs['files'][at:at]=added
            (path/name).write_text(json.dumps(theirs,indent=2)+'\n')
            git('add',name,cwd=path)
            finish(number,path,['tests/python/test_weighted_eri_binary_identity.py','tests/python/test_wb97mv_rsh_cpu_aot.py','tests/python/test_stationary_cpu_derivative_aot.py'])
        elif number == 1448:
            name='tests/python/test_program_region_boundary_validation.py'
            assert set(conflicts) <= {name}, conflicts
            own=show(EXPECTED[number],name)
            theirs=show(master,name)
            def functions(text):
                return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
            assert functions(own)==functions(theirs), 'Parent regression function semantics differ'
            (path/name).write_text(theirs)
            git('add',name,cwd=path)
            runpy.run_path(str(ROOT/'.github/scripts/astra-fix-xc-binding.py'))['repair'](path)
            names=['python/vibeqc_compiler/dft/xc_program.py','python/vibeqc_compiler/dft/xc_schedule.py','tests/python/test_dft_xc_program_region.py']
            lint=run(str(Path(sys.executable).parent/'ruff'),'check','--fix',*names,cwd=path,check=False)
            fmt=run(str(Path(sys.executable).parent/'ruff'),'format',*names,cwd=path,check=False)
            lint2=run(str(Path(sys.executable).parent/'ruff'),'check',*names,cwd=path,check=False)
            (OUT/'1448-lint.txt').write_text(lint.stdout+fmt.stdout+lint2.stdout)
            if lint2.returncode: raise RuntimeError(lint2.stdout)
            (OUT/'1448-callers.txt').write_text(git('grep','-n','-E','(bind_native_ks_xc_region_candidates|select_native_ks_xc_region_program|native_ks_host_unfused_xc_program)\(',cwd=path).stdout)
            finish(number,path,['tests/python/test_dft_xc_program_region.py','tests/python/test_program_ir_region.py','tests/python/test_program_region_boundary_validation.py','tests/python/test_xc_schedule.py'],extra_paths=['python/vibeqc_compiler/dft/xc_schedule.py'])
        else:
            # Inspect rather than guess a numerical conflict resolution.
            name='src/xtb/native/src/backends/cuda/gfn2_d4.cu'
            (OUT/'1444-vs-parent.diff').write_text(git('diff','9d13c19491b121de6e417af7a0415483b180a245',EXPECTED[number],'--',name).stdout)
    except Exception as exc:
        summary['prs'].setdefault(str(number),{})['error']=str(exc)
        (OUT/f'{number}-error.txt').write_text(traceback.format_exc())
(OUT/'round3-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
