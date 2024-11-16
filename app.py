import streamlit as st
from transformers import pipeline

# Set up the Streamlit app title and subtitle
st.title("Keyword Extraction using BERT - NER")
st.subheader("This task aims at extracting key entities from the sentence")

# Cache the model pipeline to improve efficiency
@st.cache_resource(show_spinner=True)
def load_pipe(model_ckpt):
    # Create the token-classification pipeline (removed invalid argument)
    pipe = pipeline("token-classification", model=model_ckpt)
    return pipe

# Load the token classification model
token_classification_pipe = load_pipe("ShivuuGenieExpl302001/bert-ner-custom")

# Function to perform inference using the model pipeline
def inference(text_ipt):
    return token_classification_pipe(text_ipt)

# Create a text input box in the Streamlit app for user input
text_input = st.text_input("Enter text")

# If the button is clicked, perform inference and display results
if st.button("Extract Entities"):
    st.write(inference(text_input))  # Display the extracted entities


