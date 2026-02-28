import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")

    model = load_model(os.path.join(model_dir, "lstm_model.h5"))

    with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_dir, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()
index_word = getattr(tokenizer, "index_word", None) or {
    index: word for word, index in tokenizer.word_index.items()
}


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_next_word(text):
    cleaned = text.strip()
    if not cleaned:
        return ""

    sequences = tokenizer.texts_to_sequences([cleaned])
    if not sequences or not sequences[0]:
        return ""

    sequence = pad_sequences(sequences, maxlen=max_len - 1, padding="pre")

    preds = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(preds, axis=-1)[0])
    if predicted_index == 0:
        return ""

    return index_word.get(predicted_index, "")


# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("🧠 Real-Time LSTM Next Word Prediction")

user_input = st.text_input(
    "Start typing:",
    placeholder="Type something..."
)

# Automatically runs on every keystroke
suggestion = ""

if user_input:
    suggestion = predict_next_word(user_input)

# Ghost-style suggestion
if user_input and suggestion:
    st.markdown(
        f"""
        <div style="font-size:18px; font-family:monospace; margin-top:-10px;">
            {user_input}
            <span style="color:#bbb;">
                {" " + suggestion}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )