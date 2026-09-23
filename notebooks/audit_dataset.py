"""Audit cached training Arrow shards only; no network or evaluation reads."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "huggingface_cache"


def inspect_batch(table):
    """Return scalar rows and counters without expanding all hourly data in RAM."""
    arrays = ["hours_sale", "hours_stock_status"]
    frame = table.drop(arrays).to_pandas()
    counts = Counter()
    for name in table.column_names:
        counts[f"null:{name}"] = table[name].null_count
    for name in frame.select_dtypes(include="number"):
        counts[f"nonfinite:{name}"] = int((~np.isfinite(frame[name])).sum())
    counts["invalid_dates"] = int(pd.to_datetime(frame.dt, errors="coerce").isna().sum())
    counts["negative_daily_sales"] = int((frame.sale_amount < 0).sum())
    counts["invalid_stockout_count"] = int((~frame.stock_hour6_22_cnt.between(0, 16)).sum())
    for name in ["holiday_flag", "activity_flag"]:
        counts[f"invalid:{name}"] = int((~frame[name].isin([0, 1])).sum())
    counts["discount_outside_0_1"] = int((~frame.discount.between(0, 1)).sum())
    matrices = {}
    masks = {}
    for name in arrays:
        col = table[name].combine_chunks()
        lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
        valid = lengths == 24
        counts[f"invalid_length:{name}"] = int((~valid).sum())
        flat = pc.list_flatten(col)
        counts[f"null_elements:{name}"] = flat.null_count
        values = flat.to_numpy(zero_copy_only=False)
        counts[f"nonfinite_elements:{name}"] = int((~np.isfinite(values)).sum())
        if name == "hours_sale":
            counts["negative_hourly_sales"] = int((values < 0).sum())
        else:
            counts["invalid_stock_flags"] = int((~np.isin(values, [0, 1])).sum())
        matrices[name] = pc.list_flatten(pc.filter(col, pa.array(valid))).to_numpy(
            zero_copy_only=False
        ).reshape(-1, 24)
        masks[name] = valid
    valid = masks["hours_sale"]
    counts["daily_hourly_sales_mismatch"] = int((~np.isclose(
        matrices["hours_sale"].sum(axis=1), frame.sale_amount.to_numpy()[valid],
        atol=1e-6, rtol=1e-6,
    )).sum())
    valid = masks["hours_stock_status"]
    stock = matrices["hours_stock_status"]
    recorded = frame.stock_hour6_22_cnt.to_numpy()[valid]
    # Compare both conventions explicitly rather than silently assuming flag polarity.
    counts["stock_count_mismatch_ones_06_22"] = int((stock[:, 6:22].sum(axis=1) != recorded).sum())
    counts["stock_count_mismatch_zeros_06_22"] = int(((stock[:, 6:22] == 0).sum(axis=1) != recorded).sum())
    counts["stockout_hours_06_22"] = int((stock[:, 6:22] == 1).sum())
    counts["hours_checked_06_22"] = len(stock) * 16
    if np.array_equal(masks["hours_sale"], valid):
        counts["positive_sales_hours_flagged_stockout"] = int((
            (matrices["hours_sale"] > 0) & (stock == 1)
        ).sum())
    return frame[["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]], counts


def summarize_series(frame):
    frame = frame.copy()
    frame["dt"] = pd.to_datetime(frame.dt, errors="coerce")
    keys = ["store_id", "product_id", "dt"]
    duplicates = int(frame.duplicated(keys).sum())
    unique = frame.dropna(subset=keys).drop_duplicates(keys)
    groups = unique.groupby(["store_id", "product_id"]).dt.agg(["min", "max", "count"])
    gaps = (groups["max"] - groups["min"]).dt.days + 1 - groups["count"]
    return {
        "rows": len(frame), "stores": int(frame.store_id.nunique()),
        "products": int(frame.product_id.nunique()), "store_product_series": len(groups),
        "date_min": str(unique.dt.min().date()), "date_max": str(unique.dt.max().date()),
        "duplicate_key_rows_beyond_first": duplicates,
        "series_with_internal_missing_dates": int((gaps > 0).sum()),
        "internal_missing_dates": int(gaps.sum()),
        "observations_per_series": {str(k): int(v) for k, v in groups["count"].value_counts().sort_index().items()},
        "series_start_dates": {str(k.date()): int(v) for k, v in groups["min"].value_counts().items()},
        "series_end_dates": {str(k.date()): int(v) for k, v in groups["max"].value_counts().items()},
        "sales_quantiles": {str(k): float(v) for k, v in frame.sale_amount.quantile([0, .25, .5, .75, .95, .99, 1]).items()},
        "zero_sales_by_stockout": {
            label: {"rows": int(mask.sum()), "zero_sales_rows": int((mask & frame.sale_amount.eq(0)).sum())}
            for label, mask in {
                "no_stockout_06_22": frame.stock_hour6_22_cnt.eq(0),
                "some_stockout_06_22": frame.stock_hour6_22_cnt.gt(0),
            }.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args()
    shards = sorted(args.cache_dir.rglob("*-train-*-of-*.arrow"))
    if not shards or len({p.parent for p in shards}) != 1:
        raise SystemExit("Expected one cached training revision. Run inspect_dataset.py first or specify --cache-dir.")
    expected = int(shards[0].stem.rsplit("-", 1)[1])
    if len(shards) != expected:
        raise SystemExit("Incomplete training shard set.")
    totals, frames, sources, discount_examples = Counter(), [], [], []
    for path in shards:
        print(f"Auditing {path.name}", flush=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append({"file": path.name, "sha256": digest})
        with pa.memory_map(str(path), "r") as source:
            for batch in pa.ipc.open_stream(source):
                table = pa.Table.from_batches([batch])
                if len(discount_examples) < 40:
                    discounts = table.select(["store_id", "product_id", "dt", "discount"]).to_pandas()
                    discount_examples.extend(discounts.loc[
                        ~discounts.discount.between(0, 1)
                    ].head(40 - len(discount_examples)).to_dict("records"))
                frame, counts = inspect_batch(table)
                frames.append(frame)
                totals.update(counts)
    report = {
        "dataset": "Dingdong-Inc/FreshRetailNet-50K", "split": "train",
        "cache_revision": shards[0].parent.name, "sources": sources,
        "summary": summarize_series(pd.concat(frames, ignore_index=True)),
        "checks": dict(sorted(totals.items())),
        "discount_outside_0_1_examples": discount_examples,
        "method": "Local training shards only. Sales tolerance atol=rtol=1e-6; stock window indices [6:22]. Missing dates counted within each series span; endpoints reported separately. No cleaning or imputation performed.",
    }
    out = ROOT / "reports" / "training_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Nonzero checks:", {k: v for k, v in totals.items() if v})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
