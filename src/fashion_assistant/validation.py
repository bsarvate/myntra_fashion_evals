"""Objective checks use confirmed settings and source records, never LLM prices."""
from dataclasses import dataclass, field

from .catalog import matches, positive_amount


@dataclass
class Settings:
    budget: str
    sizes: dict
    price_unit: str = "CATALOG_UNITS"
    filters: dict = field(default_factory=dict)  # per-slot confirmed hard requirements

    def __post_init__(self):
        positive_amount(self.budget)
        if set(self.sizes) not in ({"top", "bottom"}, {"set"}):
            raise ValueError("Confirm top/bottom sizes or a set size")
        self.sizes = {k: str(v).strip().upper() for k, v in self.sizes.items()}
        if not all(self.sizes.values()) or not self.price_unit:
            raise ValueError("Size and price unit are required")
        if not set(self.filters) <= self.sizes.keys():
            raise ValueError("Filter slot outside the confirmed outfit")
        for slot, filters in self.filters.items():
            matches({}, filters)  # validate field names even if no catalog matches
            if "slot" in filters and filters["slot"] != slot:
                raise ValueError("Conflicting slot filter")


def validate_outfit(ids, products, inventory, settings, inspected):
    catalog = {p["product_id"]: p for p in products}
    errors, items, total = [], [], positive_amount(settings.budget) * 0
    if len(ids) != len(set(ids)):
        errors.append("DUPLICATE_ID")
    if not set(ids) <= set(inspected):
        errors.append("INSPECTION_REQUIRED")
    for pid in ids:
        p = catalog.get(pid)
        if p is None:
            errors.append("UNKNOWN_ID")
            continue
        if p["price_unit"] != settings.price_unit:
            errors.append("PRICE_UNIT_MISMATCH")
        if not matches(p, settings.filters.get(p["slot"], {})):
            errors.append("CONSTRAINT_VIOLATION")
        size = settings.sizes.get(p["slot"])
        quantity = inventory.get("products", {}).get(pid, {}).get(size)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            errors.append("STOCK_UNKNOWN")
        elif quantity == 0:
            errors.append("OUT_OF_STOCK")
        try:
            total += positive_amount(p["price"])
        except ValueError:
            errors.append("INVALID_PRICE")
        items.append({"product_id": pid, "name": p["name"], "slot": p["slot"], "size": size,
                      "price": p["price"], "image_path": p.get("image_path", "")})
    if sorted(p["slot"] for p in items) != sorted(settings.sizes):
        errors.append("OUTFIT_STRUCTURE")
    if total > positive_amount(settings.budget):
        errors.append("OVER_BUDGET")
    if inventory.get("mode") != "SIMULATED":
        errors.append("UNSUPPORTED_INVENTORY_MODE")
    return {"valid": not errors, "errors": sorted(set(errors)), "items": items,
            "subtotal": str(total), "price_unit": settings.price_unit, "inventory_mode": inventory.get("mode"),
            "limitations": "Historical catalog prices; simulated inventory; style and fit are not validated."}
