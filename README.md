# 🧠 PsyCare-Multimodal

**AI-Powered Multimodal Mental Health Risk Predictor for College Students**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://psycare-multimodal.streamlit.app/)

PsyCare-Multimodal is an AI-driven screening tool that combines **student survey data** with **natural language analysis** to estimate mental health risk levels. It fuses predictions from traditional machine learning models (for tabular data) and a fine-tuned DistilBERT transformer (for text) into a single, interpretable risk score.

> ⚠️ **Disclaimer:** This tool is for **educational and screening purposes only** — it is **not** a medical diagnosis. If you or someone you know is in crisis, please contact a mental health professional or local emergency services immediately.

---

## 🚀 Live Demo

**[https://psycare-multimodal.streamlit.app/](https://psycare-multimodal.streamlit.app/)**

---

## ✨ Features

- **Multimodal Analysis** — Combines structured survey responses and free-text chat messages for a holistic assessment.
- **7-Class Mental Health Classification** — Detects Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, and Suicidal states from text.
- **Safety-First Design** — Keyword-based suicidal risk override ensures high-risk cases are never underestimated.
- **Severity-Aware Score Fusion** — Weighted combination of tabular risk and text severity for a balanced final score.
- **Confidence Visualization** — Interactive bar charts display prediction probabilities across all mental health classes.
- **Risk-Based Guidance** — Personalized recommendations based on the computed risk level (Low / Moderate / High).
- **Cached Model Loading** — Uses Streamlit's `@st.cache_resource` for fast, efficient inference after initial load.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Interactive web application framework |
| [PyTorch](https://pytorch.org/) | Deep learning runtime for the transformer model |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Pre-trained DistilBERT model for text classification |
| [scikit-learn](https://scikit-learn.org/) | Traditional ML models for tabular predictions |
| [joblib](https://joblib.readthedocs.io/) | Model serialization and loading |
| [NumPy](https://numpy.org/) | Numerical computations |
| [pandas](https://pandas.pydata.org/) | Tabular data manipulation |

---

## 📁 Directory Structure

```
PsyCare-Multimodal/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── tabular_models/                 # Pre-trained scikit-learn models
│   ├── depression_model.pkl        #   Depression risk classifier
│   ├── anxiety_model.pkl           #   Anxiety risk classifier
│   └── panic_model.pkl             #   Panic attack risk classifier
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

> **Note:** The DistilBERT text classification model is hosted remotely on the [Hugging Face Hub](https://huggingface.co/ultimatrix2/psycare-distilbert) and downloaded automatically at runtime.

---

## ⚙️ How It Works

### 1. Tabular Input (Student Survey)
The user provides structured data — gender, age, course, year, CGPA, marital status, treatment status, and self-reported symptoms. Three separate scikit-learn classifiers predict the probability of **depression**, **anxiety**, and **panic attacks**:

```
Tabular Risk = 0.40 × P(depression) + 0.35 × P(anxiety) + 0.25 × P(panic)
```

### 2. Text Input (Chat Message Analysis)
A fine-tuned **DistilBERT** model ([`ultimatrix2/psycare-distilbert`](https://huggingface.co/ultimatrix2/psycare-distilbert)) classifies free-text messages into one of seven mental health states:
- Anxiety, Bipolar, Depression, Normal, Personality Disorder, Stress, Suicidal

### 3. Safety Override
A keyword-based check scans the input text for suicidal risk indicators. If detected, the prediction is immediately overridden to **Suicidal** regardless of the model output.

### 4. Score Fusion
The tabular risk score and text severity weight are combined into a final risk score:

```
Final Score = 0.60 × Tabular Risk + 0.40 × Severity Weight
```

If suicidal risk is detected, the final score is forced to **1.0** (maximum risk).

### 5. Risk Classification & Guidance
| Score Range | Level | Action |
|---|---|---|
| < 0.33 | 🟢 Low Risk | Maintain healthy habits |
| 0.33 – 0.66 | 🟡 Moderate Risk | Consider speaking with a counselor |
| ≥ 0.66 | 🔴 High Risk | Professional help strongly recommended |

---

## 🏗️ Installation & Local Setup

### Prerequisites
- Python 3.8 or higher

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ultimatrix2/PsyCare-Multimodal.git
   cd PsyCare-Multimodal
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deployment

The application is deployed on **[Streamlit Community Cloud](https://streamlit.io/cloud)**.

**Live URL:** [https://psycare-multimodal.streamlit.app/](https://psycare-multimodal.streamlit.app/)

To deploy your own instance:
1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select the repository, branch, and `app.py` as the main file.
4. Click **Deploy**.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## 📄 License

This project is open source. Please check the repository for license details.

---

## 📬 Contact

For questions, suggestions, or feedback, please open an issue on the [GitHub repository](https://github.com/ultimatrix2/PsyCare-Multimodal).
