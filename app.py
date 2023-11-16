import streamlit as st
from transformers import pipeline


model = pipeline("text-classification")

def classify(text):
    result = model(text)[0]
    label = result["label"]
    score = round(result["score"], 4)
    return f"{label} (Score: {score})"

# Streamlit app
st.title("Classify Your Texts")
st.sidebar.write("This AI model is trained to classify texts/articles/sentences into four categories: World, Sports, Business, and Science/Tech.")
text_input = st.text_area("Enter your text here:")
if st.button("Classify"):
    if text_input:
        result = classify(text_input)
        st.write(result)
    else:
        st.warning("Please enter text for classification.")
