# import streamlit as st
# import pickle
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     # Get the directory where this script is located
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_dir = os.path.join(base_dir, "model")
    
#     model = load_model(os.path.join(model_dir, "lstm_model.h5"))
#     with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
#         tokenizer = pickle.load(f)
#     with open(os.path.join(model_dir, "max_len.pkl"), "rb") as f:
#         max_len = pickle.load(f)
#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_resources()
# index_word = {index: word for word, index in tokenizer.word_index.items()}

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds)

#     return index_word.get(predicted_index, "")

# def predict_top_k(text, k=3):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)[0]
#     top_indices = np.argsort(preds)[-k:][::-1]
#     return [index_word.get(idx, "") for idx in top_indices if index_word.get(idx, "")]

# def complete_sentence(text, num_words=10):
#     current_text = text.strip()
#     for _ in range(num_words):
#         next_word = predict_next_word(current_text)
#         if not next_word:
#             break
#         current_text = f"{current_text} {next_word}" if current_text else next_word
#     return current_text

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Type normally. Press 'Tab' button to accept suggestion.")

# if "suggestion" not in st.session_state:
#     st.session_state.suggestion = ""

# def update_suggestion():
#     text = st.session_state.user_input
#     if text.strip():
#         next_word = predict_next_word(text)
#         st.session_state.suggestion = next_word
#     else:
#         st.session_state.suggestion = ""

# st.text_input(
#     "✍️ Start typing:",
#     key="user_input",
#     on_change=update_suggestion,
#     placeholder="Type a sentence..."
# )

# # Show inline-style suggestion
# if st.session_state.suggestion:
#     st.markdown(
#         f"""
#         <div style='font-size:18px; color:gray; margin-top:-10px;'>
#         {st.session_state.user_input}
#         <span style='opacity:0.4;'> {st.session_state.suggestion}</span>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # Accept suggestion (VS Code style Tab simulation)
# if st.session_state.suggestion:
#     if st.button("↹ Accept Suggestion"):
#         st.session_state.user_input += " " + st.session_state.suggestion
#         st.session_state.suggestion = ""
#         st.experimental_rerun()

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
index_word = {index: word for word, index in tokenizer.word_index.items()}


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

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