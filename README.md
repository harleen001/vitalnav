# VitalNav — Jalandhar's AI Health Navigator

A production-grade Streamlit healthcare dashboard localised to **Jalandhar, Punjab**,
inspired by Bajaj Finserv Health. AI symptom triage powered by Groq
(`llama-3.3-70b-versatile`).

## Features

- **AI Symptom Check** — chat with an AI assistant that returns the top 3
  likely conditions, each with confidence %, urgency, recommended specialty
  and red-flag warnings. Uses your saved health profile for sharper triage.
- **Find Doctors** — 20 verified Jalandhar specialists with real
  qualifications, hospital affiliation, languages, fee and next available slot.
  Filter by condition, specialty, or hospital.
- **Hospitals** — 8 multi-specialty hospitals (NHS, Patel, Sacred Heart,
  CareBest, Shrimann, Oxford, Swastik Gastro Care, Sarvodya) with bed counts,
  established year, top specialties, cashless insurer network and highlights.
- **Health Plans** — IRDAI-regulated insurance plans from 8 insurers with
  explicit covered / not-covered lists, sum insured, monthly + annual price,
  IRDAI UIN, and a *Compare* mode for side-by-side analysis of up to 3 plans.
- **Cost Estimator** — indicative price ranges for 12 common procedures
  (Angioplasty, CABG, Knee Replacement, Cataract, etc.) with hospital stay
  duration, recommended Jalandhar hospitals, and insurance coverage hint.
- **My Dashboard** — consolidated view of your appointments, active plans,
  health profile snapshot and full assessment history with timestamps.
- **Personal Health Profile** — age, sex, conditions, medications, allergies
  — sent as context to the AI on every assessment.

## Local Setup (VS Code)

### 1. Prerequisites

- Python 3.11 or higher
- A free Groq API key — get one at https://console.groq.com/keys

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

**On macOS / Linux:**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**On Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

Alternatively, create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
(then reference it in code via `st.secrets["GROQ_API_KEY"]` if you prefer).

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default. To use port 5000:

```bash
streamlit run app.py --server.port 5000
```

## Project Structure

```
vitalnav/
├── app.py                       # Main Streamlit application
├── style.css                    # Bajaj-inspired custom CSS (Inter font)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   └── jalandhar_data.json      # Hospitals, doctors, plans, procedures
└── .streamlit/
    └── config.toml              # Streamlit theme + server config
```

## Data Sources

All Jalandhar hospital and doctor data was compiled from publicly listed
sources (Bajaj Finserv Health hospital directory, threebestrated.in,
hospital websites). Insurance plans reference IRDAI-listed product UINs.

> **Disclaimer:** VitalNav is for triage and information purposes only.
> It does not replace professional medical advice. In an emergency, call **108**.
