"""Build the 120-case source-derived benchmark; no model API requests."""
import json
from pathlib import Path
from fashion_assistant.benchmark120 import build_suite

root = Path(__file__).resolve().parents[1]
print(json.dumps(build_suite(root/'data/processed/myntra1000',root/'evals/myntra1000_v1'),indent=2))
