import os
import sys
import csv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

INPUT_CSV = Path("data/processed/conversation_only.csv")
OUTPUT_CSV = Path("data/processed/conversation_1000_processed.csv")
ENV_FILE = Path("env/.env")

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You rewrite only the first two turns of a clinical doctor–patient dialogue.\n"
    "Target style:\n"
    "Doctor: Good morning, Mr. [LastName]. I'm Dr. [LastName]. Let's confirm your personal details. "
    "You live at [Street, City, State], correct? And your birthdate is [Month Day, Year]?\n"
    "Patient: Yes, that's correct. [Then restate the original complaint from the input.]\n"
    "Use fake and fictitious names, State addresses, and birthdates.\n"
    "Do not change anything beyond the first two turns.\n"
)

def split_first_two_turns(text: str):
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return text.strip(), ""
    first_two = "\n".join(lines[:2]).strip()
    rest = "\n".join(lines[2:]).strip()
    return first_two, rest

def call_model(first_two: str) -> str:
    user_msg = (
        "Original first two turns:\n"
        f"{first_two}\n\n"
        "Rewrite ONLY these two turns in the style above. Output ONLY the two rewritten turns."
    )
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.6,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f_in, \
         OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or ["conversation"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            if limit and idx > limit:
                break

            conv = row["conversation"]
            first_two, rest = split_first_two_turns(conv)

            try:
                rewritten = call_model(first_two)
            except Exception as e:
                print(f"[error] row {idx}: {e}")
                rewritten = first_two

            if rest:
                final_conv = f"{rewritten}\n{rest}"
            else:
                final_conv = rewritten

            row["conversation"] = final_conv
            writer.writerow(row)
            print(f"processed {idx}")

    print(f"saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
