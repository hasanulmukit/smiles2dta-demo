# SMILES2DTA(DTC)\_Demo

🔬 **SMILES2DTA(DTC)\_Demo** is a Streamlit web application that predicts drug-target binding affinity using a trained deep learning model. It processes drug SMILES strings and protein sequences as inputs and provides the predicted binding affinity.

---

## Related Publication

This project is based on the following published research paper:

**SMILES2DTA: a CNN-based approach for identifying drug candidates and predicting drug-target binding affinity**  
[Hasanul Mukit, Sayeed Hossain, Mirza Milan Farabi, Mehrab Zaman Chowdhury, Ahmed Iqbal Pritom & Humayan Kabir Rana]  
[Neural Computing & Applications by Springer], [2024].  
Link: [https://doi.org/10.1007/s00521-024-10814-x]

Please check the publication for a detailed explanation of the model and methodology.

## Features

- Accepts drug SMILES strings and protein sequences as inputs.
- Predicts binding affinity using a trained CNN model.
- User-friendly interface with clean and modern design.

---

## How It Works

- The user enters the drug's SMILES string and the protein sequence.
- The app tokenizes the inputs, pads them to a fixed length, and passes them to the trained model.
- The predicted binding affinity is displayed in the app.

## Technologies Used

- Jupyter Notebook
- Python
- TensorFlow/Keras
- Streamlit
- Pickle

---

### Acknowledgments

- This dashboard was created as part of a research project to simplify and improve drug-target binding affinity prediction.
