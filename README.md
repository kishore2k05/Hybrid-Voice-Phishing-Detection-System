# 🛡️ Hybrid Vishing Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-96%25-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Rust-4%25-orange?logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Random%20Forest-green" />
  <img src="https://img.shields.io/badge/OCR-Tesseract-yellow" />
  <img src="https://img.shields.io/badge/Languages-English%20%7C%20Tanglish%20%7C%20Hinglish-purple" />

</p>

> A dual-layer, multilingual defense system against **Voice Phishing (Vishing)** attacks — combining Computer Vision (OCR) for technical manipulation detection with Natural Language Processing (NLP) for scam content analysis. Uniquely tuned for **Indian multilingual scenarios**, including Tanglish (Tamil-English) and Hinglish.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Innovation](#-key-innovation)
- [System Architecture](#️-system-architecture)
  - [Layer 1 — VENTINEL (Visual Guard)](#layer-1--ventinel-visual-guard)
  - [Layer 2 — NLP Classifier (Content Brain)](#layer-2--nlp-classifier-content-brain)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

---

## 🔍 Overview

**Vishing (Voice Phishing)** is a rapidly growing form of social engineering fraud where attackers impersonate banks, government agencies, or trusted contacts over phone calls to extract sensitive information or money.

Traditional vishing detection systems suffer from two major limitations:

1. They rely solely on static **phone number blacklists**, which attackers can easily bypass.
2. They are trained only on **English**, making them ineffective in multilingual countries like India, where scammers freely mix Tamil, Hindi, and English in conversation.

This project solves both problems with a **Hybrid Architecture** that operates on two independent but complementary fronts, trained on **1,584 real-world dialogue samples** spanning **4 language types** and **33 scam/service categories** with a **3-class output** (Neutral / Slightly Suspicious / Scam):

| Defense Layer | Component | Method | Role |
|---|---|---|---|
| **Layer 1** | NLP Classifier | Random Forest / BERT / Hybrid Ensemble | Classifies transcript as Neutral / Suspicious / Scam |
| **Layer 2** | Gemini Module | Google Gemini LLM | Explains *why* a call is suspicious in natural language |

---

## 💡 Key Innovation

Unlike existing vishing detection tools, this system introduces:

- **Audio-first pipeline:** Calls are transcribed directly from `.wav` recordings via the speech module, enabling end-to-end detection from raw audio without manual input.
- **Multilingual NLP:** The classifier is trained on **Tamil-English (Tanglish)** and **Hindi-English (Hinglish)** code-mixed data — making it relevant to millions of Indian users.
- **Broken-English detection:** The model also handles poorly-structured English typical of overseas scam call centers, a class rarely addressed in existing datasets.
- **3-class graded output:** Rather than a binary safe/scam decision, the system outputs a nuanced **Neutral / Slightly Suspicious / Scam** label, allowing users and downstream systems to act proportionally.
- **Multiple model variants:** Three classifier architectures (Random Forest, BERT, Hybrid Ensemble) are provided so users can benchmark and select the best fit.
- **LLM-powered explanation:** The Gemini module goes beyond a label to explain *why* a call is suspicious in plain language.
- **33 scam categories:** Coverage spans everything from classic `bank_fraud` and `lottery_scam` to emerging threats like `deepfake_voice_scam`, `ai_investment_scam`, and `pig_butchering_scam`.
- **No blacklist dependency:** The system analyzes *behavior and language*, not phone numbers, making it robust against number spoofing.

---

## 🏗️ System Architecture

```
Incoming Call Audio (.wav)
          │
          ▼
┌─────────────────────────────────┐
│     speech_module.py            │
│   (ASR — Audio Transcription)   │
│                                 │
│  .wav ──────► Raw Transcript    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     text_cleaner/ (Rust)        │
│  (Normalization & Tokenization) │
│                                 │
│  Handles: English, Tanglish,    │
│  Hinglish, Broken-English       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     LAYER 1: NLP CLASSIFIER     │
│  (Random Forest / BERT / Hybrid)│
│                                 │
│  TF-IDF / BERT Embeddings       │
│       │                         │
│  ┌────┴──────────────────┐      │
│  │ 0: Neutral            │      │
│  │ 1: Slightly Suspicious│      │
│  │ 2: Scam 🔴            │      │
│  └───────────────────────┘      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     LAYER 2: GEMINI MODULE      │
│  (LLM Explanation & Deep        │
│   Contextual Analysis)          │
│                                 │
│  Natural language reasoning on  │
│  why a call is suspicious       │
└─────────────────────────────────┘
```

---

### Layer 1 — NLP Classifier

**Goal:** Detect scam intent in the cleaned conversation transcript.

#### Models
Three model variants are implemented and can be compared:
- **`bert_model.py`** — Fine-tuned BERT transformer for sequence classification
- **`hybrid_model_soft.py`** — Soft voting ensemble combining multiple classifiers
- **`hybrid_model_stacked.py`** — Stacked ensemble with a meta-learner on top

#### Pipeline

```
Cleaned Transcript
      │
      ▼
  TF-IDF / BERT Feature Extraction
      │
      ▼
  Random Forest / BERT / Hybrid Classifier
      │
      ▼
  Label: 0 (Neutral) | 1 (Slightly Suspicious) | 2 (Scam)
```

#### Output Labels (3-Class)

| Label | Class | Meaning |
|---|---|---|
| `0` | Neutral | Legitimate service call — no threat detected |
| `1` | Slightly Suspicious | Subtle scam indicators present — proceed with caution |
| `2` | Scam | Confirmed fraudulent dialogue — high confidence |

#### Languages Supported
- 🇬🇧 **English** — Standard international and Indian English scam scripts
- 🇮🇳 **Tamil-English (Tanglish)** — Code-mixed (e.g., *"Sir, unga account block aagidum, ungaloda OTP sollunga"*)
- 🇮🇳 **Hindi-English (Hinglish)** — Code-mixed (e.g., *"Aapka lottery lag gaya hai, abhi account details do"*)
- 🔤 **Broken-English** — Poorly structured English typical of overseas scam callers

---

### Layer 2 — Gemini Module (LLM Explanation)

**Goal:** Provide a human-readable, contextual explanation of *why* a transcript is suspicious.

`gemini_module.py` sends the cleaned transcript to the **Google Gemini API**, which returns a natural language breakdown of suspicious patterns — useful for user-facing alerts, auditing, or further investigation beyond the classifier's label.

---

## 📊 Dataset

The project uses two dataset files, both located in `datasets/`. Together they represent a rich, multi-class, multilingual corpus of real-world vishing dialogues.

---

### `dataset_fixed_v3_cleaned.csv` — Core Dataset

**1,065 labeled dialogue samples** across 4 languages and 27 scam/service categories.

**Language Breakdown:**

| Language | Samples | Description |
|---|---|---|
| **English** | 456 | Standard international and Indian English scam scripts |
| **Hindi-English (Hinglish)** | 256 | Hindi-English code-mixed Indian scam dialogues |
| **Tamil-English (Tanglish)** | 252 | Tamil-English code-mixed scam dialogues — a key innovation |
| **Broken-English** | 101 | Poorly structured English typical of overseas scam callers |

**Label Distribution (3-Class):**

| Label | Description | Count |
|---|---|---|
| `0` | Neutral — legitimate service call | 282 |
| `1` | Slightly Suspicious — subtle scam indicators | 176 |
| `2` | Scam — confirmed fraudulent dialogue | 607 |

**Scam / Service Categories (27 types):**

| Category | Samples | Category | Samples |
|---|---|---|---|
| `bank_fraud` | 166 | `social_engineering` | 45 |
| `investment_fraud` | 97 | `refund_scam` | 43 |
| `genuine_service` | 75 | `govt_scheme` | 42 |
| `tech_support` | 60 | `genuine_bank` | 39 |
| `digital_arrest_scam` | 56 | `tax_fraud` | 35 |
| `lottery_scam` | 54 | `job_scam` | 33 |
| `hard_negative` | 53 | `delivery_service` | 31 |
| `it_support` | 26 | `sim_block_scam` | 24 |
| `credit_card_offer` | 22 | `insurance_fraud` | 21 |
| `telecom_service` | 20 | `survey_call` | 20 |
| `donation_scam` | 19 | `utility_scam` | 18 |
| `insurance_call` | 16 | `medical_service` | 14 |
| `delivery_scam` | 13 | `loan_scam` | 13 |
| `task_scam` | 10 | | |

---

### `dataset_full_combined_cleaned.csv` — Full Augmented Training Dataset

**1,584 samples** — the complete training corpus, including original samples plus augmented data. This is the primary file used for model training.

**Columns:** `id`, `dialogue`, `scam_type`, `language`, `label`, `label_description`, `source`, `split`

**Language Breakdown:**

| Language | Samples |
|---|---|
| **English** | 584 |
| **Tamil-English (Tanglish)** | 421 |
| **Hindi-English (Hinglish)** | 412 |
| **Broken-English** | 167 |

**Label Distribution:**

| Label | Description | Count |
|---|---|---|
| `0` | Neutral | 455 |
| `1` | Slightly Suspicious | 507 |
| `2` | Scam | 622 |

**Data Sources (How Samples Were Generated):**

| Source | Samples | Description |
|---|---|---|
| `augmented_train` | 1,100 | Back-translation and paraphrase augmentation |
| `original` | 177 | Hand-crafted original dialogues |
| `enhancement_phase2` | 169 | Second-phase quality enhancement samples |
| `keyword_balance` | 49 | Added to balance underrepresented scam keywords |
| `asr_noise` | 41 | Samples with simulated ASR transcription noise |
| `new_content` | 31 | Freshly authored dialogues for coverage gaps |
| `subtle_scam` | 17 | Deliberately subtle scam scripts for hard cases |

**Notable Scam Categories (33 types, including emerging threats):**

In addition to all categories from v3, the full dataset adds: `banking_fraud`, `deepfake_voice_scam`, `ai_investment_scam`, `pig_butchering_scam`, and `job_fraud` — reflecting newer, more sophisticated attack vectors.

> 💡 All 1,584 samples are assigned to the `train` split. Evaluation is performed via cross-validation on held-out subsets.

**Dataset Location:** `datasets/`

---

## 📂 Project Structure

```
Hybrid-Vishing-Detection-System/
│
├── Models/                             # ML model scripts and evaluation outputs
│   ├── saved_models/                   # Serialized trained model artifacts
│   ├── test_results/
│   │   └── results.csv                 # Model evaluation results
│   ├── bert_model.py                   # BERT-based classifier implementation
│   ├── hybrid_model_soft.py            # Hybrid model with soft voting ensemble
│   ├── hybrid_model_stacked.py         # Hybrid model with stacked ensemble
│   ├── test_model.py                   # Model testing and inference script
│   └── train_model.py                  # Model training pipeline
│
├── Modules/                            # Core runtime modules
│   ├── gemini_module.py                # Gemini API integration for LLM-based analysis
│   ├── speech_module.py                # Speech-to-text / audio processing
│   └── translation_module.py          # Language detection and translation utilities
│
├── datasets/                           # All data assets
│   ├── audio_files/                    # Raw audio call recordings
│   │   ├── Microsoft Support Call Scam Audio Recording - Lyas Spiehler.wav
│   │   ├── normal_01.wav ~ normal_04.wav   # Legitimate call samples
│   │   └── scam_01.wav ~ scam_03.wav       # Scam call samples
│   ├── cleaned_transcripts/            # Preprocessed transcripts (after text_cleaner)
│   │   ├── Microsoft Support Call Scam Audio Recording - Lyas Spiehler.txt
│   │   ├── normal_01.txt ~ normal_04.txt
│   │   └── scam_01.txt ~ scam_03.txt
│   ├── raw_transcripts/                # Raw ASR-generated transcripts
│   │   ├── Microsoft Support Call Scam Audio Recording - Lyas Spiehler.txt
│   │   ├── normal_01.txt ~ normal_04.txt
│   │   └── scam_01.txt ~ scam_03.txt
│   ├── dataset_fixed_v3.csv            # Core dataset (1,065 samples, 4 languages)
│   └── dataset_full_combined.csv       # Full augmented training set (1,584 samples)
│
├── test_results/
│   └── results.csv                     # Top-level evaluation results
│
├── text_cleaner/                       # Rust-based text preprocessing module
│   ├── src/
│   │   └── main.rs                     # Rust entry point (tokenization, normalization)
│   ├── Cargo.lock
│   └── Cargo.toml
│
├── Cargo.lock                          # Root Rust dependency lockfile
├── Cargo.toml                          # Root Rust workspace config
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites

Ensure the following are installed on your system:

- **Python 3.8+**
- **Rust** (required for the `text_cleaner` module)
- **pip** (Python package manager)
- **Google Gemini API Key** (required for `gemini_module.py`)

#### Installing Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/kishore2k05/Hybrid-Vishing-Detection-System.git
cd Hybrid-Vishing-Detection-System
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Key dependencies include:** `scikit-learn`, `transformers`, `torch`, `pandas`, `numpy`, `nltk`, `google-generativeai`, `SpeechRecognition`, `deep-translator`.

### Step 4: Build the Rust Text Cleaner

```bash
cd text_cleaner
cargo build --release
cd ..
```

### Step 5: Configure the Gemini API Key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

Or export it directly in your shell:

```bash
export GEMINI_API_KEY=your_api_key_here
```

---

## 🚀 Usage

### Step 1 — Transcribe an Audio Call

Use `speech_module.py` to convert a `.wav` call recording into a text transcript.

```bash
python Modules/speech_module.py --audio datasets/audio_files/scam_01.wav --output datasets/raw_transcripts/scam_01.txt
```

---

### Step 2 — Clean the Transcript

The Rust-based `text_cleaner` normalizes and tokenizes the raw transcript, handling code-mixed languages.

```bash
cd text_cleaner
cargo run -- --input ../datasets/raw_transcripts/scam_01.txt --output ../datasets/cleaned_transcripts/scam_01.txt
```

---

### Step 3 — Classify with the NLP Model

Run the trained model on a cleaned transcript to get a scam prediction.

```bash
python Models/test_model.py --transcript datasets/cleaned_transcripts/scam_01.txt
```

**Example output:**

```
[Classifier] Input Language: English
[Classifier] Processing transcript...
[Classifier] 🔴  Label: 2 — SCAM (Confidence: 94.2%)
[Classifier] Suspicious patterns: OTP solicitation, urgency trigger, account block threat
```

---

### Step 4 (Optional) — Gemini LLM Analysis

For a deeper natural language explanation of why a call is suspicious, use the Gemini module.

```bash
python Modules/gemini_module.py --transcript datasets/cleaned_transcripts/scam_01.txt
```

---

### Training the Model from Scratch

```bash
python Models/train_model.py --dataset datasets/dataset_full_combined.csv --output Models/saved_models/
```

To train the BERT-based model specifically:

```bash
python Models/bert_model.py --dataset datasets/dataset_full_combined.csv --output Models/saved_models/
```

---

## 📈 Model Performance

Results from evaluation runs are stored in `test_results/`. Summary of classifier performance on the held-out test set:

| Metric | Score |
|---|---|
| **Accuracy** | ~95%+ |
| **Precision** | High (minimizes false alarms) |
| **Recall** | High (minimizes missed scams) |
| **F1-Score** | Balanced across all three language classes |

> 📝 Exact metrics may vary based on train/test split. Run the evaluation script to reproduce results:
> ```bash
> python Modules/nlp_classifier/evaluate.py --model Models/ --dataset datasets/test_set.csv
> ```

---

## 🧰 Technologies Used

| Category | Technology |
|---|---|
| **Language (Primary)** | Python 3.8+ |
| **Language (Text Cleaner)** | Rust |
| **Machine Learning** | scikit-learn (Random Forest), BERT, Hybrid Ensemble (Soft Voting & Stacked) |
| **LLM Integration** | Google Gemini API (`gemini_module.py`) |
| **Speech-to-Text** | `speech_module.py` (ASR pipeline for audio transcription) |
| **NLP / Text Processing** | NLTK, custom Rust-based `text_cleaner` module |
| **Feature Extraction** | TF-IDF Vectorizer, BERT embeddings |
| **Translation / Detection** | `translation_module.py` (language detection + back-translation) |
| **Data Handling** | pandas, NumPy |
| **Build System (Rust)** | Cargo |

---

## 👥 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests where applicable
4. Commit your changes: `git commit -m "feat: describe your change"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a **Pull Request** with a clear description of changes

### Areas Where Contributions Are Welcome

- Adding more Tanglish / regional language training samples
- Improving OCR accuracy for varied screen formats
- Experimenting with transformer-based models (BERT, MuRIL) for the NLP layer
- Real-time audio transcription integration (Whisper, Vosk)
- Building a mobile companion app

### Reporting Issues

If you encounter bugs or have feature suggestions, please [open an issue](https://github.com/kishore2k05/Hybrid-Vishing-Detection-System/issues) with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your OS and Python version

---

## 🙏 Acknowledgements

- **scikit-learn** — Machine learning library for Python
- **Hugging Face Transformers** — BERT model implementation
- **Google Gemini** — LLM-powered contextual analysis
- **Dravidian CodeMix NLP research community** — For pioneering work on Tamil-English code-mixed text analysis
- All contributors who helped build and test this system

---

## 👨‍💻 Authors

| Name | GitHub |
|---|---|
| **Kishore Gowthaman** | [@kishore2k05](https://github.com/kishore2k05) |
| **Sri Varshini B S** | [@sri0024](https://github.com/sri0024) |

---

<p align="center">
  Built with ❤️ to make phone calls safer for everyone — especially in multilingual India.
</p>
