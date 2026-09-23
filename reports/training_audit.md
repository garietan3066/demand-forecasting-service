# FreshRetailNet training-data audit

Audited all cached training rows on 2026-09-23. Evaluation files were not opened. No records were cleaned, dropped, or imputed. Detailed counts, training-shard SHA-256 hashes, cache revision, and discount examples are in [training_audit.json](training_audit.json).

## Coverage and integrity

| Check | Result |
|---|---:|
| Daily records | 4,500,000 |
| Stores / products | 898 / 865 |
| Store-product series | 50,000 |
| Coverage for every series | 2024-03-28 through 2024-06-25 |
| Unique days per series | 90 |
| Duplicate store-product-date records beyond first | 0 |
| Missing dates within series | 0 |
| Null values in any column or hourly element | 0 |
| Nonfinite numeric values | 0 |
| Negative daily or hourly sales | 0 |
| Hourly arrays with length other than 24 | 0 |
| Daily/hourly sales sum mismatches | 0 |
| Invalid binary stockout flags | 0 |
| Invalid holiday/activity flags | 0 |

Sales totals use absolute and relative tolerances of 1e-6. Data is processed in Arrow batches to avoid expanding all hourly arrays into a large pandas table.

## Stockout interpretation

Counting hourly flags equal to **1** at indices `[6:22]` matches `stock_hour6_22_cnt` on every row. Thus the released data uses **1 = stockout**, **0 = not flagged as stockout**. Counting zero flags instead disagrees on 4,388,182 rows. The window covers 16 hourly bins starting at 06:00 through 21:00; it excludes the bin starting at 22:00.

- 1,992,006 days (44.27%) have at least one flagged stockout hour in this window.
- 14,311,536 of 72,000,000 window-hours (19.88%) are flagged as stockouts.
- Days without stockouts in this window: 2,507,994; zero-sales days among them: 46,944 (1.87%).
- Days with stockouts in this window: 1,992,006; zero-sales days among them: 153,816 (7.72%).

The hourly percentage and daily percentage have different denominators; they should not be confused. No-stockout in this window does not establish availability for the other eight hours.

Across all 24 hours, **772,426 hourly observations have positive sales and a stockout flag**. These must not automatically be treated as corrupt records or overwritten with zero. A plausible explanation is sales followed by a stockout within the same hour, but exact within-hour flag timing is not established by this audit. Verify this before using masks to reconstruct demand.

## Values needing interpretation

There are **34 discount values outside [0, 1]**. Examples include 1.07 and 1.088. This is a review flag, not proof of corruption: the field may encode a price ratio above the reference price. Retain the original values pending a documented interpretation. Examples are saved in the JSON report.

Median daily sales are 0.7, the 95th percentile is 2.9, and the maximum is 44.9, all in the publisher's normalized scale. These are not original physical unit counts.

## Modeling decisions before training

1. Preserve raw data and stockout flags. Do not equate stockout zero-sales observations with known zero demand.
2. Start with a deterministic subset selected without evaluation outcomes, and establish a chronological validation period within training history.
3. Calculate lag features within each store-product series using only past observations. Compare against a seven-day seasonal naive baseline.
4. Evaluate observed-sales forecasting separately from demand-recovery experiments. Actual unmet demand during real stockouts is not provided as ground truth. Tests that hide in-stock observations provide a controlled experiment, not proof of real lost-demand accuracy.
5. Never use the target day's realized stockout status or realized weather as ordinary inputs to a forecast made the previous day. Historical stockouts can inform features and label handling; future covariates require forecast-time availability.
6. With only 90 training days, prioritize short-term and weekday effects. Do not claim annual seasonality was learned.
7. Treat any replenishment demonstration as a simulation with stated inventory and lead-time assumptions.

## Reproduce

First run the existing inspection script to populate the cache. It may download both splits, but the audit reads only training Arrow shards:

```powershell
.\venv\Scripts\python.exe notebooks/inspect_dataset.py
.\venv\Scripts\python.exe notebooks/audit_dataset.py
.\venv\Scripts\python.exe -m pytest -q
```

If multiple cached revisions exist, pass `--cache-dir` pointing to the intended revision folder. The script refuses ambiguous or incomplete shard sets. Generated reports contain aggregate findings and a small set of encoded-ID examples, not the raw dataset.

Source: [Dingdong-Inc FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), CC BY 4.0. Dataset authors: Yangyang Wang, Jiawei Gu, Li Long, Xin Li, Li Shen, Zhouyu Fu, Xiangjun Zhou, and Xu Jiang. [Dataset paper](https://arxiv.org/abs/2505.16319). Published field descriptions establish normalization and the stockout-count window; numerical findings above come from this local audit.
