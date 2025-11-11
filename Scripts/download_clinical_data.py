from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
ds = load_dataset("AGBonnet/augmented-clinical-notes")

print(ds)

data = ds["train"]
df = pd.DataFrame(data)

print(df.head())

df.to_csv("clinical_notes_raw.csv", index=False, encoding="utf-8")
print("Saved to clinical_notes_raw.csv")
