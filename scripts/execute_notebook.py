"""Execute default notebook cells using the current Python, saving visible outputs."""
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--sys-prefix',
                       '--name', 'fashion-assistant', '--display-name', 'Fashion Assistant (Python 3.11)'])
notebook = nbformat.read(root/'notebooks/01_build_and_evaluate.ipynb', as_version=4)
nbformat.validate(notebook)
# Refuse to inadvertently run a locally edited notebook with external operations enabled.
source = '\n'.join(''.join(cell.source) for cell in notebook.cells if cell.cell_type == 'code')
for flag in ('USE_REAL_DATA', 'ENABLE_MODEL_DOWNLOADS', 'ENABLE_PINECONE', 'ENABLE_LIVE_LLM', 'RUN_LIVE_BENCHMARK', 'MOUNT_DRIVE'):
    if f'{flag} = False' not in source or f'{flag} = True' in source:
        raise ValueError(f'Offline verification requires {flag}=False. Use interactive cells for other runs.')
NotebookClient(notebook, timeout=180, kernel_name='fashion-assistant',
               resources={'metadata': {'path': str(root)}}).execute()
output = root/'artifacts/01_build_and_evaluate.executed.ipynb'
output.parent.mkdir(exist_ok=True)
nbformat.write(notebook, output)
print('Executed all default notebook cells:', output)
