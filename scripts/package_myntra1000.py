"""Bundle only the prepared subset for Colab, leaving the raw download separate."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
subset = root/'data/processed/myntra1000'
if not (subset/'catalog.json').is_file():
    raise RuntimeError('Run prepare_myntra1000.py first')
output = root/'dist/myntra1000-data.zip'
output.parent.mkdir(exist_ok=True)
with ZipFile(output,'w',ZIP_DEFLATED) as archive:
    for path in sorted(subset.rglob('*')):
        if path.is_file() and path.name != '.DS_Store':
            archive.write(path,Path('myntra1000')/path.relative_to(subset))
print(output, f'({output.stat().st_size/1024**2:.1f} MiB)')
