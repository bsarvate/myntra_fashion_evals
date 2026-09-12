"""Explicit CSV mapping and auditable rejection; no guessed image-to-product joins."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = ("product_id", "name", "description", "slot", "category", "color", "fabric", "price", "price_unit", "image_path")
FILTER_FIELDS = {"slot", "category", "color", "fabric"}


def positive_amount(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a price")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid amount") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("Amount must be positive and finite")
    return amount


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inspect_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {"columns": reader.fieldnames, "rows": len(rows), "preview": rows[:3],
                "malformed_rows": sum(None in r or any(v is None for v in r.values()) for r in rows)}


def prepare_catalog(path, mapping, *, category_slots=None, image_root=None, price_unit=None):
    """mapping: canonical name -> actual CSV column. Missing facts stay empty.

    category_slots is an explicit reviewed map; no substring garment inference.
    Paths must be relative to image_root. Missing/broken images retain text records.
    """
    category_slots = category_slots or {}
    products, rejected, image_issues = [], [], []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(mapping.values()) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapped columns absent from CSV: {sorted(missing)}")
        if not {"product_id", "name", "price"} <= mapping.keys():
            raise ValueError("Map product_id, name and price explicitly")
        for line, row in enumerate(reader, start=2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Malformed CSV row; repair source copy before importing")
                p = {key: (row.get(mapping.get(key, ""), "") or "").strip() for key in FIELDS}
                if not p["product_id"] or not p["name"]:
                    raise ValueError("Missing product_id/name")
                if p["product_id"] in seen:
                    raise ValueError("Duplicate product ID")
                p["price"] = str(positive_amount(p["price"]))
                p["slot"] = p["slot"].lower() or category_slots.get(p["category"], "")
                if p["slot"] not in {"top", "bottom", "set"}:
                    raise ValueError("Missing/unsupported slot; provide a reviewed category_slots map")
                p["price_unit"] = p["price_unit"] or price_unit or "CATALOG_UNITS"
                p["image_verified"] = False
                if p["image_path"]:
                    try:
                        if image_root is None:
                            raise ValueError("No image root configured")
                        root = Path(image_root).resolve()
                        image_path = (root / p["image_path"]).resolve()
                        if Path(p["image_path"]).is_absolute() or not image_path.is_relative_to(root):
                            raise ValueError("Image must be relative to image_root")
                        from PIL import Image
                        with Image.open(image_path) as image:
                            image.verify()
                        p["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                        p["image_verified"] = True
                    except (ValueError, OSError) as exc:
                        image_issues.append({"product_id": p["product_id"], "reason": str(exc)})
                else:
                    image_issues.append({"product_id": p["product_id"], "reason": "No image mapping"})
                seen.add(p["product_id"])
                products.append(p)
            except ValueError as exc:
                rejected.append({"line": line, "reason": str(exc)})
    report = {"accepted": len(products), "rejected": rejected, "image_issues": image_issues,
              "slots": dict(Counter(p["slot"] for p in products)), "catalog_sha256": fingerprint(products),
              "note": "Image verification checks files, not semantic correctness of the product-image join."}
    return products, report


def matches(product, filters):
    unknown = set(filters) - FILTER_FIELDS
    if unknown:
        raise ValueError(f"Unsupported filters: {sorted(unknown)}")
    return all(str(product.get(k, "")).casefold() == str(v).strip().casefold() for k, v in filters.items())


def product_text(product):
    return " | ".join(str(product.get(k, "")) for k in ("name", "description", "category", "color", "fabric", "slot"))
