"""Continue the isolated review with strict, deletion-only D4 validation."""
from pathlib import Path

harness = Path('.github/scripts/astra-review-1444-1453.py').read_text()
harness = harness.replace('for number in (1452, 1448, 1444):', 'for number in (1448, 1444):')
harness = harness.replace("'tests/python/test_xc_schedule.py'", "'tests/python/test_dft_xc_schedule.py','tests/python/test_schedule_identity_admission.py','tests/python/test_schedule_contract.py','tests/python/test_common_precision.py'")
harness = harness.replace("'round3-summary.json'", "'round4-summary.json'")
old = '''        else:
            # Inspect rather than guess a numerical conflict resolution.
            name='src/xtb/native/src/backends/cuda/gfn2_d4.cu'
            (OUT/'1444-vs-parent.diff').write_text(git('diff','9d13c19491b121de6e417af7a0415483b180a245',EXPECTED[number],'--',name).stdout)
'''
new = r'''        else:
            import difflib
            import hashlib
            import re
            name='src/xtb/native/src/backends/cuda/gfn2_d4.cu'
            manifest_name='src/xtb/native/CUDA_SOURCE_PROVENANCE.json'
            assert set(conflicts) <= {name,manifest_name}, conflicts
            text=(path/name).read_text()
            block=re.compile(r'(?m)^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n',re.S)
            removed=[]
            def resolve(match):
                ours, theirs=match.groups()
                assert not ours.strip(), 'Unexpected nonempty dense-delete conflict side'
                assert 'Gfn2D4DeviceCache' in theirs, 'Unexpected live conflict'
                removed.append(theirs)
                return ours
            repaired,count=block.subn(resolve,text)
            assert count >= 1, 'Expected explicit dead-cache delete/modify conflicts'
            assert not re.search(r'(?m)^(<<<<<<<|=======|>>>>>>>)',repaired)
            (path/name).write_text(repaired)
            git('add',name,cwd=path)
            def tokens(text):
                text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
                return re.findall(r'"(?:\\.|[^"\\])*"|\w+|[^\s]',text)
            # The integration must only delete existing tokens, not alter math,
            # live traversal, guards, publication, or branch conditions.
            audit=[]
            for source in (name,name+'h'):
                current=tokens(show(master,source))
                candidate=tokens((path/source).read_text())
                cursor=iter(current)
                assert all(any(item==token for item in cursor) for token in candidate), source
                audit.append(f'{source}: deletion-only token subsequence verified; {len(current)} -> {len(candidate)} tokens')
            retired=('Gfn2D4DeviceCache','update_gfn2_d4_geometry_cache_cuda','evaluate_gfn2_d4_two_body_cuda','evaluate_gfn2_d4_scc_potential_cuda','evaluate_gfn2_d4_scc_energy_cuda','add_gfn2_d4_two_body_gradient_cuda','evaluate_gfn2_d4_atm_cuda','add_gfn2_d4_atm_gradient_cuda')
            references=git('grep','-n','-E','|'.join(retired),'--','src','include',cwd=path,check=False)
            assert references.returncode == 1, references.stdout
            audit.append('No retained production references to retired dense API or cache type')
            manifest=json.loads(show(master,manifest_name))
            original=json.loads(show(EXPECTED[number],manifest_name))
            manifest['adaptations']['src/backends/cuda/gfn2_d4.cu']=original['adaptations']['src/backends/cuda/gfn2_d4.cu']
            for source in ('src/backends/cuda/gfn2_d4.cu','src/backends/cuda/gfn2_d4.cuh'):
                manifest['files'][source]['vendored_sha256']=hashlib.sha256((path/'src/xtb/native'/source).read_bytes()).hexdigest()
            (path/manifest_name).write_text(json.dumps(manifest,indent=2)+'\n')
            git('add',manifest_name,cwd=path)
            (OUT/'1444-deletion-audit.txt').write_text('\n'.join(audit)+'\n')
            finish(number,path,['tests/python/test_gfn2_cuda_provenance.py'])
'''
# The raw replacement contains regex escapes for the compiled harness itself.
new = new.replace('\\\\', '\\')
assert old in harness
harness = harness.replace(old,new)
exec(compile(harness, '.github/scripts/astra-review-1444-1453.py', 'exec'))
