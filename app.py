import streamlit as st
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from helper_funcs import clean_txt, text_to_wordlist
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Comment Toxicity Detection",page_icon="💬",layout="wide")

model_path = hf_hub_download(repo_id="allabilitiessrank/biLSTM", filename="bilstm_mini.keras")
model = load_model(model_path)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Insights",
        "Real-time Prediction",
        "Bulk Prediction"
    ]
)

if menu == "Home":

    st.title("**Deep Learning for Comment Toxicity Detection**")

    st.write("""
    This application predicts whether an online comment is toxic.

    **Framework**: TensorFlow + Streamlit
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training Sentences", "159,571")
    col2.metric( "Test Samples", "153,164")
    col3.metric( "Vocabulary", "100,000")
    col4.metric("Model", "BiLSTM")

    d1, d2 = st.columns(2)
    d1.metric("Accuracy","0.98")
    d2.metric("Val Accuracy","0.97")

elif menu == "Data Insights":

    st.title("Dataset Insights")
    train = pd.read_csv("Data/train.csv")

    st.write("Dataset Shape")
    st.dataframe(pd.DataFrame({"Rows":[train.shape[0]],"Columns":[train.shape[1]]}))

    labels = ["toxic", "severe_toxic","obscene", "threat", "insult","identity_hate"]

    st.subheader("Label Distribution")

    st.bar_chart(train[labels].sum())

elif menu == "Real-time Prediction":

    st.title("Predict Comment Toxicity")
    comment = st.text_area("Enter Comment")

    if st.button("Predict"):
        comment = clean_txt(comment)
        comment = text_to_wordlist(comment)

        seq = tokenizer.texts_to_sequences([comment])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        pred = model.predict(pad)[0]

        st.subheader("Prediction")

        for label, score in zip(labels, pred):

            st.write(f"**{label}** : {score:.2%}")
            st.progress(float(score))


elif menu == "Bulk Prediction":

    st.title("Bulk Prediction")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file is not None:

        df = pd.read_csv(file)
        if "comment_text" not in df.columns:
            st.error("CSV must contain comment_text column")
        else:
            clean = df["comment_text"].apply(clean_txt)

            seq = tokenizer.texts_to_sequences(clean)

            pad = pad_sequences(seq,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            pred = model.predict(pad)

            df['toxic'] = pred[:,0]
            df['severe_toxic'] = pred[:,1]
            df['obscene'] = pred[:,2]
            df['threat'] = pred[:,3]
            df['insult'] = pred[:,4]
            df['identity_hate'] = pred[:,5]

            st.dataframe(df.head())

            csv = df.to_csv(index=False)

            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")