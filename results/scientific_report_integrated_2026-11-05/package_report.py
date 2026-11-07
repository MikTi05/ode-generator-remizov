"""Package the completed report and smoke-test its standalone saved-only checks."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
excluded = {'report_bundle.zip', 'bundle_manifest.json', 'package_verification.json', 'FILE_INDEX.md'}
files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in excluded
               and '__pycache__' not in p.parts)
index = '# Файлы новой редакции\n\nВсе пути относительно этого каталога. '
index += 'Исторические оригиналы не включены в список изменённых файлов.\n\n'
index += '\n'.join('- `'+str(p.relative_to(HERE))+'`' for p in files)
index += '\n\nДополнительно: `bundle_manifest.json`, `report_bundle.zip`, `package_verification.json`.\n'
(HERE/'FILE_INDEX.md').write_text(index)
files.append(HERE/'FILE_INDEX.md')
manifest = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(HERE/'bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
files.append(HERE/'bundle_manifest.json')
archive = HERE/'report_bundle.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
    for path in sorted(files):
        entry = zipfile.ZipInfo(str(path.relative_to(HERE)),date_time=(2026,11,5,0,0,0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o100644 << 16
        bundle.writestr(entry,path.read_bytes())
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name,digest in manifest.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == digest,name
    with tempfile.TemporaryDirectory(prefix='2488-report-unpack-') as temporary:
        bundle.extractall(temporary)
        runs = []
        for script in ('verify_report.py','verify_numeric_saved.py'):
            run = subprocess.run([sys.executable,'-B',str(Path(temporary)/script)],
                                 text=True,capture_output=True,check=True)
            runs.append(dict(script=script,exit_code=run.returncode))
result = dict(status='passed',archive=archive.name,entries=len(files),
              archive_bytes=archive.stat().st_size,
              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              CRC='passed',SHA256_checked_files=len(manifest),standalone_checks=runs,
              new_solver_MC_ML_calls=0)
(HERE/'package_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
