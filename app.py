import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set Streamlit page config
st.set_page_config(page_title="SMILES2DTA (Demo)", page_icon="🔬", layout="centered")

# Load your trained model
model = load_model('mDTBA_v1dtc.h5', compile=False)

# Load tokenizers
with open('tokenizer_smi.pkl', 'rb') as file:
    tokenizer_smi = pickle.load(file)

with open('tokenizer_pro.pkl', 'rb') as file:
    tokenizer_pro = pickle.load(file)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to left top, #bdff95, #6abea0);
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .header {
        color:rgb(2, 25, 63);
        font-size: 32px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        font-family: 'Poppins', sans-serif;

    }
     /* Button hover effect */
    button {
        background: linear-gradient(to left top, #16ac33, #64faec);
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        font-family: 'Poppins', sans-serif;
    }
    button:hover {
        background: linear-gradient(to left top, #43e45e, #54befd);
        transform: scale(1.05);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #000;
        font-size: 14px;
        font-family: 'Poppins', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header">🔬 SMILES2DTA (Demo):Binding Affinity Prediction</div>', unsafe_allow_html=True)

# Info Section
st.info("Predict drug-target binding affinity based on SMILES strings and target protein sequences.")

# Input fields
st.subheader("Input Data")
drug_smiles = st.text_input("Enter Drug SMILES String:")
protein_sequence = st.text_area("Enter Target Protein Sequence:")

# Prediction button
if st.button("Predict Binding Affinity"):
    if not drug_smiles or not protein_sequence:
        st.error("Please enter both SMILES string and protein sequence!")
    else:
        # Preprocess inputs
        drug_seq = tokenizer_smi.texts_to_sequences([drug_smiles])
        drug_padded = pad_sequences(drug_seq, truncating="post", padding="post", maxlen=85)

        protein_seq = tokenizer_pro.texts_to_sequences([protein_sequence])
        protein_padded = pad_sequences(protein_seq, truncating="post", padding="post", maxlen=1200)

        # Predict
        prediction = model.predict([drug_padded, protein_padded])
        predicted_affinity = prediction.item()

        # Display result
        st.success(f"Predicted Binding Affinity: {predicted_affinity:.4f}")

# Footer
st.markdown('<div class="footer">Created by Hasanul Mukit</div>', unsafe_allow_html=True)
