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
    "You rewrite ONLY the first two turns of a clinical doctor-patient dialogue.\n"
    "Output must be in ENGLISH only.\n\n"
    
    "STRICT REQUIREMENTS:\n"
    "1. Doctor's turn MUST include:\n"
    "   - Greeting with patient's name (Mr./Ms./Mrs. [LastName])\n"
    "   - Doctor's self-introduction\n"
    "   - Verification of AT LEAST 2 items from: patient name, address, birthdate\n"
    "   - MUST include birthdate/DOB in verification\n"
    "2. Patient's turn MUST:\n"
    "   - Confirm the information (brief acknowledgment)\n"
    "   - State ONLY the original complaint - no additions, no modifications\n"
    "3. Do NOT change or add anything beyond these two turns\n"
    "4. Use completely fictional names and details\n\n"
    
    "FORMAT STRUCTURE:\n"
    "Doctor: [Greeting], [Patient Title+LastName]. [Doctor introduction]. [Verify 2-3 details including birthdate]\n"
    "Patient: [Brief confirmation]. [Original complaint exactly as provided]\n\n"
    
    "VARIATION ELEMENTS:\n"
    "Greetings: Good morning/afternoon/Hello\n"
    "Doctor intros: I'm Dr./My name is Dr./I'll be your doctor today, Dr.\n"
    "Verification phrases: Let me confirm/Can you verify/Just to verify/I have here that\n"
    "Birthdate formats: birthdate/date of birth/DOB + various date formats\n"
    "Patient confirmations: Yes/That's correct/That's right/Yes, that's me\n"
    "Complaint transitions: So/Well/Actually/I'm here because/The reason I'm here is\n\n"
    
    "CORRECT EXAMPLES:\n"
    "Example 1:\n"
    "Doctor: Good morning, Ms. Anderson. I'm Dr. Mitchell. Let me confirm - you're Jennifer Anderson, born April 10, 1985, living at 456 Pine Street, Boston, Massachusetts?\n"
    "Patient: Yes, that's correct. I'm here because I'm experiencing discomfort in my neck and lower back.\n\n"
    
    "Example 2:\n"
    "Doctor: Hello, Mr. Davis. My name is Dr. Lee. Can you verify your date of birth is June 23, 1972, and you reside in Portland, Oregon?\n"
    "Patient: That's right. Well, I've been having trouble with my neck and lower back.\n\n"
    
    "CRITICAL RULES:\n"
    "- MUST verify birthdate in every dialogue\n"
    "- MUST verify at least one other detail (name or address)\n"
    "- NEVER add extra text after the patient's complaint\n"
    "- NEVER use any language other than English\n"
    "- Keep patient's original complaint EXACTLY as is\n"
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
