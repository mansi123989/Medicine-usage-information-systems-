# 💊 AI-Powered Medicine Information Assistant

## 📖 Overview
AI-Powered Medicine Information Assistant is a Streamlit-based application that extracts medicine information from package images using OCR and NLP. It predicts medicine uses using a multi-label machine learning model and provides multilingual audio explanations to improve user experience and accessibility.

## 📸 Screenshots
The **Screenshots/** folder contains sample outputs and the user interface of the application, showcasing its key features and workflow.

## 🔄 Workflow
1. Upload a medicine package image.
2. Extract text using EasyOCR.
3. Correct OCR output using Groq Llama 3.
4. Extract medicine information using NLP.
5. Predict medicine uses using the trained ML model.
6. Generate simplified medicine-use explanations.
7. Convert explanations into multilingual speech.

## 🛠️ Technologies Used
- Python
- Streamlit
- EasyOCR
- spaCy
- Scikit-learn
- Groq API (Llama 3)
- FDA API
- gTTS

## 🤖 Machine Learning
- Trained on **192K+ medicine records**.
- Multi-label classification across **566 medicine-use labels**.
- Achieved **98.2% Precision**.


