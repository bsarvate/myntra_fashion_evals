"""Versioned retrieval/trajectory evaluation. Synthetic fixtures are not benchmarks."""
import csv
import importlib.metadata
import json
import math
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .catalog import fingerprint, matches


def load_cases(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_cases(cases, products):
    catalog = {p["product_id"]: p for p in products}
    ids, groups = set(), {}
    for case in cases:
        cid = case["case_id"]
        if cid in ids:
            raise ValueError("Duplicate case_id")
        ids.add(cid)
        group, split = case["group_id"], case["split"]
        if split not in {"dev", "test"}:
            raise ValueError("Split must be dev or test")
        if group in groups and groups[group] != split:
            raise ValueError("Leakage: related cases cross dev/test splits")
        groups[group] = split
        if case.get("evidence") not in {"synthetic", "human_reviewed", "source_derived", "pending_review"}:
            raise ValueError("Declare the label evidence type")
        judgments = case["judgments"]
        if not set(judgments) <= catalog.keys():
            raise ValueError("Judgment references unknown product")
        if any(isinstance(g, bool) or g not in (0, 1, 2, 3) for g in judgments.values()):
            raise ValueError("Relevance grade must be 0, 1, 2 or 3")
        eligible = {pid for pid, p in catalog.items() if matches(p, case.get("filters", {}))}
        if any(grade > 0 and pid not in eligible for pid, grade in judgments.items()):
            raise ValueError("Relevant judgment violates a hard filter")
        if case.get("exhaustive") and not eligible <= judgments.keys():
            raise ValueError("Exhaustive cases must explicitly judge every eligible product")
        if case.get("expected_no_match") and (not case.get("exhaustive") or any(judgments.values())):
            raise ValueError("No-match truth requires exhaustive zero-relevance judgments")
    return {"cases": len(cases), "groups": len(groups), "benchmark_sha256": fingerprint(cases)}


def retrieval_metrics(ranking, case, products, k):
    if k <= 0:
        raise ValueError("k must be positive")
    ids = [row["product_id"] for row in ranking[:k]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate result IDs")
    catalog = {p["product_id"]: p for p in products}
    judgments = case["judgments"]
    relevant = {pid for pid, grade in judgments.items() if grade > 0}
    unjudged = [pid for pid in ids if pid not in judgments]
    hits = len(set(ids) & relevant)
    # Unjudged documents are not automatically scored as irrelevant.
    scoreable = not unjudged
    dcg = sum((2**judgments[pid]-1)/math.log2(rank+2) for rank, pid in enumerate(ids)) if scoreable else None
    ideal = sum((2**grade-1)/math.log2(rank+2) for rank, grade in
                enumerate(sorted(judgments.values(), reverse=True)[:k]))
    return {"returned": len(ids), "unjudged_count": len(unjudged),
            "judgment_coverage": (len(ids)-len(unjudged))/len(ids) if ids else None,
            "precision_at_k": hits/k if scoreable else None,
            "precision_returned": hits/len(ids) if ids and scoreable else None,
            "recall_at_k": hits/len(relevant) if case.get("exhaustive") and relevant else None,
            "ndcg_at_k_judged_pool": dcg/ideal if scoreable and ideal else None,
            "no_match_success": float(not ids) if case.get("expected_no_match") else None,
            "constraint_violation_rate": sum(pid not in catalog or not matches(catalog[pid], case.get("filters", {}))
                                             for pid in ids)/len(ids) if ids else None,
            "returned_ids": ids}


def bootstrap_mean(values, seed=42, samples=1000):
    if not values:
        return {"mean": None, "n": 0, "ci95": None}
    result = {"mean": statistics.mean(values), "n": len(values), "ci95": None}
    if len(values) >= 2:
        rng = random.Random(seed)
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
        result["ci95"] = [means[int(samples*.025)], means[min(int(samples*.975), samples-1)]]
    return result


def summarize(rows):
    metrics = ("precision_at_k", "precision_returned", "recall_at_k", "ndcg_at_k_judged_pool",
               "no_match_success", "constraint_violation_rate", "judgment_coverage", "latency_ms")
    output = {}
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        output[method] = {"cases": len(subset), "errors": sum(r.get("error") is not None for r in subset)}
        for metric in metrics:
            # Bootstrap independent groups, not near-duplicate paraphrases.
            groups = {}
            for row in subset:
                if row.get(metric) is not None:
                    groups.setdefault(row["group_id"], []).append(row[metric])
            output[method][metric] = bootstrap_mean([statistics.mean(v) for v in groups.values()])
    return output


def run_retrieval(cases, products, methods, *, split="dev", k=5):
    audit_cases(cases, products)
    rows, skipped = [], []
    for case in cases:
        if case["split"] != split:
            continue
        if case.get("evidence") == "pending_review":
            skipped.append({"case_id": case["case_id"], "reason": "Human relevance labels pending"})
            continue
        for method, search in methods.items():
            if method not in case.get("methods", list(methods)):
                continue
            row = {"case_id": case["case_id"], "group_id": case["group_id"], "category": case["category"],
                   "method": method, "evidence": case["evidence"], "error": None}
            start = time.perf_counter()
            try:
                row.update(retrieval_metrics(search(case, k), case, products, k))
            except Exception as exc:
                row["error"] = type(exc).__name__
            row["latency_ms"] = (time.perf_counter()-start)*1000
            rows.append(row)
    if not rows and not skipped:
        raise ValueError("No runnable cases for this split/method combination")
    return {"rows": rows, "skipped": skipped, "summary": summarize(rows),
            "by_category": {category: summarize([r for r in rows if r["category"] == category])
                            for category in sorted({r["category"] for r in rows})},
            "split": split, "k": k, "benchmark_sha256": fingerprint(cases), "catalog_sha256": fingerprint(products),
            "evidence": sorted({row["evidence"] for row in rows}),
            "interpretation": "Group-macro means; group bootstrap CI. Errors are separate, not dropped silently. "
                              "Recall requires exhaustive labels; nDCG uses judged pool. No-match scored separately."}


def save_run(report, output_dir, configuration):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / stamp
    path.mkdir()
    packages = {}
    for name in ("fashion-assistant-evals", "langchain-core", "langchain-nebius", "langgraph", "pinecone", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_root = Path(__file__).parent
    manifest = {"created_utc": stamp, "python": platform.python_version(), "packages": packages,
                "configuration": configuration,
                "source_sha256": fingerprint({p.name: p.read_text() for p in sorted(source_root.glob("*.py"))})}
    (path/"report.json").write_text(json.dumps(report, indent=2))
    (path/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def export_review_sheet(cases, rankings, output_path):
    """Blank labels: the assistant never fabricates human relevance judgments."""
    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "product_id", "reviewer", "relevance_0_3", "notes"])
        writer.writeheader()
        for case in cases:
            for pid in sorted({r["product_id"] for r in rankings.get(case["case_id"], [])}):
                writer.writerow({"case_id": case["case_id"], "product_id": pid})


def score_turn(result, expected, previous=None):
    """Objective trajectory assertions; style/claim correctness still need review."""
    proposal = result.get("proposal")
    selected = {p["product_id"] for p in proposal["items"]} if proposal else set()
    statuses = expected.get("status_any_of", [expected.get("status")])
    checks = {"outcome": result["status"] in statuses}
    if "required_ids" in expected:
        checks["required_ids"] = set(expected["required_ids"]) <= selected
    if "forbidden_ids" in expected:
        checks["forbidden_ids"] = not (set(expected["forbidden_ids"]) & selected)
    if "max_calls" in expected:
        checks["call_bound"] = result["calls"] <= expected["max_calls"]
    if proposal:
        checks["validated"] = proposal["valid"] is True
    current_slots = {p['slot']:p['product_id'] for p in proposal['items']} if proposal else {}
    previous_slots = {p['slot']:p['product_id'] for p in previous['proposal']['items']} if previous and previous.get('proposal') else {}
    for slot in expected.get('retain_slots', []):
        checks['retain_'+slot] = slot in current_slots and slot in previous_slots and current_slots[slot] == previous_slots[slot]
    for slot in expected.get('replace_slots', []):
        checks['replace_'+slot] = slot in current_slots and slot in previous_slots and current_slots[slot] != previous_slots[slot]
    for slot, allowed in expected.get('allowed_ids_by_slot', {}).items():
        checks['allowed_'+slot] = current_slots.get(slot) in allowed
    return {"checks": checks, "success": all(checks.values())}


def run_conversations(cases, agent_factory, repeats=1, *, checkpoint_path=None, progress=False, stop_on_api_error=False):
    """Fresh agent for each case/repeat; preserve state only within each dialogue."""
    if repeats < 1:
        raise ValueError("At least one repeat is required")
    case_ids, groups = set(), {}
    for case in cases:
        if case["case_id"] in case_ids:
            raise ValueError("Duplicate conversation ID")
        case_ids.add(case["case_id"])
        group, split = case["group_id"], case["split"]
        if split not in {"dev", "test"} or (group in groups and groups[group] != split):
            raise ValueError("Invalid split or conversation-group leakage")
        groups[group] = split
        if case.get("evidence") not in {"synthetic", "human_reviewed", "source_derived"} or not case["turns"]:
            raise ValueError("Declare evidence and at least one conversation turn")
    rows = []
    planned = sum(len(c['turns']) for c in cases) * repeats
    expected_lengths = {c['case_id']: len(c['turns']) for c in cases}
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists():
            raise ValueError("Checkpoint already exists; choose a new run path")

    def snapshot(status):
        dialogues = {}
        for row in rows:
            dialogues.setdefault((row['case_id'], row['repeat']), []).append(row['success'])
        complete = [v for (cid, _), v in dialogues.items() if len(v) == expected_lengths[cid]]
        api_errors = sum(r.get('result', {}).get('status') == 'api_error' for r in rows)
        execution_errors = sum(r.get('error') is not None for r in rows)
        if status == 'completed' and api_errors + execution_errors:
            status = 'all_turns_failed_to_execute' if api_errors + execution_errors == len(rows) else 'completed_with_errors'
        report = {'rows': rows, 'turn_success_rate': statistics.mean(r['success'] for r in rows) if rows else None,
                  'dialogue_success_rate': statistics.mean(all(v) for v in complete) if complete else None,
                  'dialogues': len(complete), 'incomplete_dialogues': len(dialogues)-len(complete),
                  'turns': len(rows), 'planned_turns': planned, 'repeats': repeats,
                  'api_error_turns': api_errors, 'execution_error_turns': execution_errors,
                  'run_status': status, 'benchmark_sha256': fingerprint(cases),
                  'note': 'Rates cover recorded turns and completed dialogues only. Partial runs are not full benchmark results. Expected outcomes require independent review.'}
        if checkpoint:
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(checkpoint)
        return report

    snapshot('running')
    try:
        for case in cases:
            for repeat in range(repeats):
                agent = agent_factory(case)
                previous = None
                for turn_index, turn in enumerate(case['turns']):
                    if progress:
                        print(f"[{len(rows)+1}/{planned}] {case['case_id']} repeat {repeat+1}, turn {turn_index+1}: starting", flush=True)
                    start = time.perf_counter()
                    try:
                        result = agent.chat(turn['request'], image_path=turn.get('image_path'))
                        row = {'result': result, **score_turn(result, turn['expected'], previous), 'error': None}
                        previous = result
                    except Exception as exc:
                        row = {'success': False, 'error': type(exc).__name__}
                    row.update(case_id=case['case_id'], group_id=case['group_id'], evidence=case['evidence'], repeat=repeat,
                               turn=turn_index, latency_ms=(time.perf_counter()-start)*1000)
                    rows.append(row)
                    snapshot('running')
                    if progress:
                        print(f"  {row.get('result', {}).get('status', row.get('error'))}; success={row['success']}; {row['latency_ms']/1000:.1f}s; completed turn saved={checkpoint is not None}", flush=True)
                    if stop_on_api_error and (row.get('error') or row.get('result', {}).get('status') == 'api_error'):
                        return snapshot('stopped_on_error')
    except KeyboardInterrupt:
        report = snapshot('interrupted')
        if progress:
            print(f"Interrupted. Preserved {len(rows)}/{planned} completed turns. In-flight turn is unrecorded.", flush=True)
        return report
    except Exception:
        snapshot('setup_error')
        raise
    return snapshot('completed')


def summarize_human_reviews(path):
    """Summarize submitted judgments; blanks remain missing, not perfect scores.

    Agreement pairs are matched on run/case/dimension/claim, before adjudication.
    Returns exact agreement, not an inflated claim of calibrated LLM judging.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    dimensions, factual, judgments = {}, [], {}
    for row in rows:
        if not row.get("reviewer"):
            continue
        dimension = row.get("dimension", "")
        rating = row.get("score_0_3", "").strip()
        supported = row.get("supported_yes_no", "").strip().lower()
        key = (row.get("case_id"), row.get("run_id"), dimension, row.get("claim"))
        value = None
        if rating:
            score = int(rating)
            if score not in (0, 1, 2, 3):
                raise ValueError("Human score must be 0–3")
            dimensions.setdefault(dimension, []).append(score)
            value = str(score)
        if supported:
            if supported not in {"yes", "no", "uncertain"}:
                raise ValueError("Claim support must be yes, no or uncertain")
            if not row.get("claim"):
                raise ValueError("Claim assessment requires claim text")
            factual.append(supported)
            value = supported
        if value is not None:
            reviewer = row["reviewer"]
            by_reviewer = judgments.setdefault(key, {})
            if reviewer in by_reviewer:
                raise ValueError("Duplicate human judgment for the same review unit")
            by_reviewer[reviewer] = value
    agreements = []
    for values in judgments.values():
        labels = list(values.values())
        agreements.extend(labels[i] == labels[j] for i in range(len(labels)) for j in range(i+1, len(labels)))
    assessed = [value for value in factual if value != "uncertain"]
    return {"dimension_ratings": {key: {"mean": statistics.mean(values), "n_ratings": len(values)} for key, values in dimensions.items()},
            "unsupported_claim_rating_rate": assessed.count("no")/len(assessed) if assessed else None,
            "assessed_claim_ratings": len(assessed), "uncertain_claim_ratings": factual.count("uncertain"),
            "pairwise_exact_agreement": statistics.mean(agreements) if agreements else None,
            "agreement_pairs": len(agreements),
            "note": "Counts are reviewer ratings, not unique adjudicated claims. Exact agreement is not chance-corrected."}
