MedNoteLM - Medical Dialogue Anonymization Pipeline
A pipeline for processing medical dialogues to create training datasets for PII (Personally Identifiable Information) detection and anonymization models.
Overview
This project processes clinical conversation data through a multi-step pipeline:

Downloads raw clinical notes from HuggingFace
Extracts conversation columns
Adds synthetic PII to dialogues (names, addresses, birthdates)
Redacts PII using OpenAI's API to create anonymized versions

Project Structure
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
│   ├── redact_sensitive_llm.py        # Add synthetic PII
│   └── redact_with_openai.py          # Anonymize with OpenAI
└── requirements.txt
Setup

Install dependencies:

bashpip install -r requirements.txt
```

2. **Configure API key:**
Create `env/.env` file:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o
Usage
Step 1: Download Dataset
bashpython Scripts/download_clinical_data.py
Downloads the AGBonnet/augmented-clinical-notes dataset from HuggingFace.
Step 2: Extract Conversations
bashpython Scripts/extract_conversation.py
Extracts 2000 conversation rows from the raw dataset.
Step 3: Add Synthetic PII
bashpython Scripts/redact_sensitive_llm.py
```
Adds synthetic patient names, doctor names, addresses, and birthdates to the first two dialogue turns.

Example transformation:
```
Original:
Doctor: Good morning, what brings you in today?
Patient: I have been experiencing chest pain.

With PII:
Doctor: Good morning, Mr. Smith. I'm Dr. Chen. Let's confirm your personal details. You live at 123 Main St, Boston, MA, correct? And your birthdate is January 15, 1985?
Patient: Yes, that's correct. I have been experiencing chest pain.
Step 4: Create Anonymized Version
bashpython Scripts/redact_with_openai.py
```
Replaces all PII with 'xxx' placeholders for training anonymization models.

Result:
```
Doctor: Good morning, Mr. xxx. I'm Dr. xxx. Let's confirm your personal details. You live at xxx, correct? And your birthdate is xxx?
Patient: Yes, that's correct. I have been experiencing chest pain.
Dataset Details

Source: AGBonnet/augmented-clinical-notes (HuggingFace)
Size: 1000-2000 conversations
Processing: Only first two dialogue turns are modified
PII Types: Names, addresses, birthdates, medical record numbers

API Usage Notes

Uses OpenAI API (gpt-4o-mini recommended for cost efficiency)
Processing 1000 conversations takes ~2-3 hours
Rate limiting: 0.4s delay between calls

Output Files

conversation_1000_processed.csv: Conversations with synthetic PII added
conversation_1000_redacted.csv: Anonymized version for model training

Requirements

Python 3.8+
OpenAI API key
~8GB disk space for datasets

License
Research use only. Please cite the original AGBonnet/augmented-clinical-notes dataset.