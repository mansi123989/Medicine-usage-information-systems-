# 💊 AI-Powered Medicine Information Assistant

## 📖 Overview
AI-Powered Medicine Information Assistant is a Streamlit-based application that extracts medicine information from package images using OCR and NLP. It predicts medicine uses using a multi-label machine learning model and provides multilingual audio explanations to improve user experience and accessibility.

## 🎯 Target Users

- Patients seeking quick and easy-to-understand medicine information.
- Elderly and illiterate users who cannot easily search or read medicine information online, benefiting from multilingual audio explanations.

  

## 🚀 Live Demo

Coming Soon...

## 📸 Screenshots
The **project_demo/** folder contains sample outputs and the user interface of the application, showcasing its key features and workflow.

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

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mansi123989/Medicine-usage-information-systems-.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Medicine-usage-information-systems-
   ```

3. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   **Windows**
   ```bash
   venv\Scripts\activate
   ```

   **Linux/macOS**
   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run main_app.py
   ```

6. Open the URL displayed in the terminal (usually **http://localhost:8501**) in your web browser.
