# MedNoteLM - Medical Dialogue Anonymization Pipeline

A pipeline for processing medical dialogues to create training datasets for PII (Personally Identifiable Information) detection and anonymization models.

---

## Overview

This project processes clinical conversation data through a multi-step pipeline:

1. Downloads raw clinical notes from HuggingFace
2. Extracts conversation columns
3. Adds synthetic PII to dialogues (names, addresses, birthdates)
4. Redacts PII using OpenAI's API to create anonymized versions

---

## Project Structure

```
MedNoteLM/
├── data/
│   ├── raw/
│   │   └── clinical_notes_raw.csv       # Original dataset
│   └── processed/
│       ├── conversation_only.csv        # Extracted conversations
│       ├── conversation_1000_processed.csv  # With added PII
│       └── conversation_1000_redacted.csv   # Anonymized version
├── env/
│   └── .env                             # API keys
├── Scripts/
│   ├── download_clinical_data.py       # Download from HuggingFace
│   ├── extract_conversation.py         # Extract conversation column
│   ├── redact_sensitive_llm.py         # Add synthetic PII
│   └── redact_with_openai.py           # Anonymize with OpenAI
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MedNoteLM.git
cd MedNoteLM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create `env/.env` file:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o
```

---

## Usage

### Step 1: Download Dataset

```bash
python Scripts/download_clinical_data.py
```

from datasets import load_dataset
ds = load_dataset("AGBonnet/augmented-clinical-notes")

### Step 2: Extract Conversations

```bash
python Scripts/extract_conversation.py
```

Extracts 2000 conversation rows from the raw dataset.

### Step 3: Add Synthetic PII

```bash
python Scripts/redact_sensitive_llm.py
```

Adds synthetic patient names, doctor names, addresses, and birthdates to the first two dialogue turns.

#### Example transformation:

**Before:**

```
Doctor: Good morning, what brings you in today?
Patient: I have been experiencing chest pain.
```

**After:**

```
Doctor: Good morning, Mr. Smith. I'm Dr. Chen. Let's confirm your personal details.
You live at 123 Main St, Boston, MA, correct? And your birthdate is January 15, 1985?
Patient: Yes, that's correct. I have been experiencing chest pain.
```

### Step 4: Create Anonymized Version

```bash
python Scripts/redact_with_openai.py
```

Replaces all PII with 'xxx' placeholders for training anonymization models.

#### Result:

```
Doctor: Good morning, Mr. xxx. I'm Dr. xxx. Let's confirm your personal details.
You live at xxx, correct? And your birthdate is xxx?
Patient: Yes, that's correct. I have been experiencing chest pain.
```

---

## Dataset Details

* **Source**: AGBonnet/augmented-clinical-notes (HuggingFace)
* **Size**: 1000–2000 conversations
* **Processing**: Only first two dialogue turns are modified
* **PII Types**: Names, addresses, birthdates, medical record numbers

---

## API Usage Notes

* Uses OpenAI API (**gpt-4o-mini** recommended for cost efficiency)
* Processing 1000 conversations takes ~2–3 hours
* Rate limiting: 0.4s delay between calls
* Estimated cost: ~$5–10 for 1000 conversations

---

## Output Files

| File                              | Description                            |
| --------------------------------- | -------------------------------------- |
| `conversation_1000_processed.csv` | Conversations with synthetic PII added |
| `conversation_1000_redacted.csv`  | Anonymized version for model training  |

---

## Requirements

* Python 3.8+
* OpenAI API key
* ~8GB disk space for datasets
* 4GB RAM minimum

---

## Dependencies

```
pandas>=1.3.0
openai>=1.0.0
python-dotenv>=0.19.0
tqdm>=4.65.0
datasets>=2.0.0
pathlib
```

---

## License

Research use only. Please cite the original AGBonnet/augmented-clinical-notes dataset.

---

## Citation

```
@dataset{augmented_clinical_notes,
  title={Augmented Clinical Notes},
  author={AGBonnet},
  year={2024},
  publisher={HuggingFace}
}
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For questions or issues, please open an issue on GitHub.
