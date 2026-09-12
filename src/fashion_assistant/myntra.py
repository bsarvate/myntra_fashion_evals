"""Deterministic, stratified subset with explicit source and image provenance."""
import ast
import csv
import hashlib
import html
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .catalog import positive_amount, fingerprint

QUOTAS = {"top": 400, "jeans": 150, "shorts": 100, "trousers": 100, "skirt": 50, "set": 200}
SEED = "myntra-v1-seed42"


def plain(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text))).strip()


def classify(name, attributes):
    # Source-declared top+bottom combination takes precedence over title tokens.
    if attributes.get("Top Type") not in (None, "NA", "") and attributes.get("Bottom Type") not in (None, "NA", ""):
        return "set", "set"
    title = name.casefold()
    # Exclude multipacks and uncertain multi-piece titles from single-garment rules.
    if re.search(r"\b(pack|set|with|jumpsuit|dress|kurta|lehenga)\b", title):
        return None
    for token, category in [(r"jeans", "jeans"), (r"shorts", "shorts"), (r"trousers", "trousers"), (r"skirts?", "skirt")]:
        if re.search(r"\b"+token+r"\b", title):
            return "bottom", category
    if re.search(r"\b(tops?|shirts?|blouses?)\b", title):
        return "top", "top"
    return None


def candidates(csv_path):
    rejected, unique, duplicates = Counter(), {}, []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for ordinal, row in enumerate(rows):
        try:
            numeric_id = Decimal(row["p_id"])
            if not numeric_id.is_finite() or numeric_id != numeric_id.to_integral_value() or numeric_id <= 0:
                raise ValueError("Invalid product ID")
            pid = str(int(numeric_id))
            price = str(positive_amount(row["price"]))
            attributes = ast.literal_eval(row["p_attributes"])
            if not isinstance(attributes, dict):
                raise ValueError("Invalid attributes")
        except (ValueError, SyntaxError, ArithmeticError):
            rejected["missing_or_invalid_core_fields"] += 1
            continue
        if pid in unique:
            duplicates.append(pid)
            continue
        category = classify(row["name"], attributes)
        if category is None:
            rejected["outside_supported_single_garment_or_source_declared_set"] += 1
            continue
        slot, label = category
        fabric = attributes.get("Fabric", "")
        if slot == "set":
            top_fabric, bottom_fabric = attributes.get("Top Fabric"), attributes.get("Bottom Fabric")
            # A single set fabric is only declared when the two source components agree.
            fabric = top_fabric if top_fabric == bottom_fabric else ""
            if attributes.get("Dupatta") == "With Dupatta" and attributes.get("Dupatta Fabric") != fabric:
                fabric = ""  # Do not label a mixed-fabric set using only its top fabric.
        if not fabric or fabric == "NA" or not row["colour"].strip():
            rejected["missing_or_mixed_required_attributes"] += 1
            continue
        url = row["img"].replace("http://", "https://", 1)
        if not url.startswith("https://assets.myntassets.com/") or f"/images/{pid}/" not in url:
            rejected["source_url_id_mismatch"] += 1
            continue
        unique[pid] = {"product_id": pid, "name": row["name"], "description": plain(row["description"]),
                       "slot": slot, "category": label, "color": row["colour"].strip(), "fabric": fabric,
                       "price": price, "price_unit": "CATALOG_UNITS", "image_path": f"images/{pid}.jpg",
                       "source_row_ordinal": ordinal, "source_csv_index": row[""], "source_image_url": url,
                       "brand": row["brand"], "source_attributes": attributes,
                       "category_provenance": "source Top Type + Bottom Type" if slot == "set" else "bounded title rule; reviewable",
                       "fabric_provenance": "source top/bottom and included dupatta fabrics agree" if slot == "set" else "source Fabric"}
    return list(unique.values()), {"raw_rows": len(rows), "eligible_unique": len(unique), "rejected_counts": dict(rejected),
                                  "duplicate_eligible_rows": len(duplicates),
                                  "csv_index_unique_values": len({row[""] for row in rows}),
                                  "note": "CSV index repeats by category and must never be used as an image join."}


def select_subset(records):
    selected = []
    for category, count in QUOTAS.items():
        pool = [p for p in records if p["category"] == category]
        pool.sort(key=lambda p: hashlib.sha256((SEED+p["product_id"]).encode()).hexdigest())
        if len(pool) < count:
            raise ValueError(f"Need {count} {category} products, found {len(pool)}")
        selected.extend(pool[:count])
    assert len(selected) == 1000 and len({p["product_id"] for p in selected}) == 1000
    return selected


def write_catalog(products, output):
    from .catalog import FIELDS
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(products)
    (output/"catalog.json").write_text(json.dumps(products, indent=2, ensure_ascii=False))
    return fingerprint(products)
