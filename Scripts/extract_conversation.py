import pandas as pd
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "raw" / "clinical_notes_raw.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "conversation_2000.csv"

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    conv_df = df[df["conversation"].notna()][["conversation"]].head(2000).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved {len(conv_df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
