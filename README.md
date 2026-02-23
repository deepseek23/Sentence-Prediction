# 🧠 Real-Time Next Word Prediction — LSTM

A deep learning-powered web application that predicts and auto-completes your sentence in real-time as you type, built with an LSTM neural network and deployed using Streamlit.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Warning](#warning)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

This project implements a **sequence-to-word** LSTM model trained on a quotes dataset. The model predicts the most likely next word based on the input sequence and displays it as a ghost/inline suggestion — similar to how Gmail's Smart Compose or IDE autocomplete works.

---

## Demo Video

Watch here: https://www.youtube.com/watch?v=bYnn0d1-7kw
---

## Features

- ⚡ Real-time next-word prediction on every keystroke
- 👻 Ghost-style inline suggestion (grayed out next word)
- 🧠 LSTM-based deep learning model
- 🗂️ Trained on a curated quotes dataset
- 🚀 Fast inference with cached model loading via `@st.cache_resource`

---

## Project Structure

```
word_prediction/
│
├── app/
│   ├── app.py                  # Streamlit application
│   ├── model/
│   │   ├── lstm_model.h5       # Trained LSTM model
│   │   ├── tokenizer.pkl       # Fitted Keras tokenizer
│   │   ├── max_len.pkl         # Saved max sequence length
│   └── notebook/
│   |    └── lstm_gru.ipynb      # Training & experimentation notebook
│   |────data/
|         ├── quotes.csv        # Training dataset      
├── .gitignore
└── README.md
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| TensorFlow / Keras | LSTM model training & inference |
| Streamlit | Web app framework |
| NumPy | Numerical operations |
| Pickle | Model artifact serialization |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/word_prediction.git
cd word_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install streamlit tensorflow numpy
```

### 4. Run the application

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ⚠️ Warning — Do NOT Train the Model Locally

> **Training this LSTM model on a local machine is strongly discouraged.**

Training a deep LSTM network on text data is **computationally expensive** and can:

- 🔥 **Overheat** your CPU/GPU and cause hardware damage
- 🐢 Take **hours** to complete on consumer-grade hardware
- 💥 **Crash your machine** due to memory exhaustion on large sequence datasets
- 🔋 Drain resources, making your machine unusable during training

### Recommended Alternatives for Training

| Platform | Notes |
|---|---|
| [Google Colab](https://colab.research.google.com/) | Free GPU/TPU — recommended |
| [Kaggle Notebooks](https://www.kaggle.com/code) | Free GPU with quota |
| [AWS / GCP / Azure](https://aws.amazon.com/) | Paid cloud compute |

The pre-trained model (`lstm_model.h5`), tokenizer (`tokenizer.pkl`), and max length (`max_len.pkl`) are already included in the `app/model/` directory. **Use them directly.**

---

## How It Works

1. User types text into the input box.
2. The input is tokenized using the pre-fitted Keras `Tokenizer`.
3. The token sequence is padded to `max_len - 1`.
4. The LSTM model outputs a probability distribution over the vocabulary.
5. The word with the highest probability (`argmax`) is selected.
6. The predicted word is displayed inline as a ghost suggestion in real-time.

```
Input Text  →  Tokenize  →  Pad Sequence  →  LSTM Model  →  Argmax  →  Predicted Word
```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute with attribution.

---


