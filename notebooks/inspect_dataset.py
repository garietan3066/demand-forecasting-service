"""Inspect training data; the loader may also download/cache evaluation files."""

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_cache"


def main():
    train = load_dataset(
        "Dingdong-Inc/FreshRetailNet-50K",
        split="train",
        cache_dir=str(CACHE_DIR),
    )

    print("\nTRAINING DATA")
    print(train)

    print("\nCOLUMN TYPES")
    print(train.features)

    # Start with a small preview rather than converting everything to pandas.
    preview = train.select(range(min(5, len(train)))).to_pandas()

    print("\nFIRST FIVE RECORDS")
    print(
        preview[
            [
                "store_id",
                "product_id",
                "dt",
                "sale_amount",
                "stock_hour6_22_cnt",
            ]
        ].to_string(index=False)
    )

    first = train[0]

    print("\nFIRST RECORD: HOURLY ARRAYS")
    print("Hourly sales:", first["hours_sale"])
    print("Hourly stockout status:", first["hours_stock_status"])
    print("Sales array length:", len(first["hours_sale"]))
    print("Stockout array length:", len(first["hours_stock_status"]))


if __name__ == "__main__":
    main()
