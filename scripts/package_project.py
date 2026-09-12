"""Create a portable source bundle, excluding credentials and user data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root / "dist" / "fashion-assistant-evals.zip"
output.parent.mkdir(exist_ok=True)
allowed_dirs = {"src", "notebooks", "scripts", "tests", "config", "evals"}
allowed_files = {"README.md", "pyproject.toml", ".env.example", ".gitignore", "BUILD_STATUS.md", "requirements-tested.txt"}
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(part in {".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"} or part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.parts[0] in allowed_dirs or str(rel) in allowed_files or rel.parts[:2] == ("data", "sample") or path.name == ".gitkeep":
            archive.write(path, Path(root.name)/rel)
print(output)
