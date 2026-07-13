# medicine_predictor.py
# Handles ML predictions and FDA API integration

import os
import requests
import numpy as np
import joblib
import streamlit as st
from typing import List


class Config:
    """Configuration for medicine prediction"""
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')  # Add this line
    FDA_API_KEY = os.getenv('FDA_API_KEY', '')
    FDA_BASE_URL = "https://api.fda.gov/drug/label.json"
    CONFIDENCE_THRESHOLD = 0.3
    MODEL_PATH = 'medicine_model.pkl'
    LABEL_ENCODER_PATH = 'label_encoder.pkl'


class MLPredictor:
    """Handles ML model predictions for medicine uses"""
    
    def __init__(self):
        self.pipeline, self.label_encoder = self._load_models()
    
    @staticmethod
    def _load_models():
        """Load the trained ML model and label encoder"""
        try:
            if os.path.exists(Config.MODEL_PATH):
                pipeline = joblib.load(Config.MODEL_PATH)
                print(f"✅ ML Model loaded from: {Config.MODEL_PATH}")
            else:
                print(f"❌ Model not found at: {Config.MODEL_PATH}")
                return None, None
            
            if os.path.exists(Config.LABEL_ENCODER_PATH):
                label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
                print(f"✅ Label encoder loaded from: {Config.LABEL_ENCODER_PATH}")
            else:
                print(f"❌ Label encoder not found at: {Config.LABEL_ENCODER_PATH}")
                return pipeline, None
            
            return pipeline, label_encoder
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_uses(self, medicine_name: str, contains: List[str] = None) -> List[str]:
        """
        Predict medicine uses using the trained model
        Input format: NAME + CONTAINS (matches training)
        """
        if self.pipeline is None or self.label_encoder is None:
            return ['ML Model not loaded. Please train the model first.']
        
        try:
            # Prepare input (NAME + CONTAINS) - matches training
            parts = []
            if medicine_name:
                parts.append(medicine_name)
            if contains:
                parts.extend(contains)
            
            input_text = " ".join(parts)
            print("ML INPUT:", input_text)
            
            # Get prediction probabilities
            probs = self.pipeline.predict_proba([input_text])[0]
            
            # Get Top 5 predictions
            top = np.argsort(probs)[-5:][::-1]
            
            predicted_labels = []
            print("\nTop Predictions:")
            
            for i in top:
                label = self.label_encoder.classes_[i]
                confidence = probs[i]
                print(f"{label} -> {confidence:.3f}")
                
                # Keep only reasonably confident predictions
                if confidence >= 0.05:
                    predicted_labels.append(label)
            
            # If nothing crosses threshold, still return top 3
            if not predicted_labels:
                predicted_labels = [
                    self.label_encoder.classes_[i]
                    for i in top[:3]
                ]
            
            print(f"📊 Predicted uses: {predicted_labels}")
            return predicted_labels
            
        except Exception as e:
            print(f"⚠️ ML Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return ['Error: Unable to predict']


class FDAApi:
    """Handles FDA API integration for medicine information"""
    
    def __init__(self):
        self.api_key = Config.FDA_API_KEY
    
    def get_medicine_uses(self, medicine_name: str) -> List[str]:
        """Fetch medicine uses from FDA API"""
        if not self.api_key:
            return []
        
        try:
            url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medicine_name}+OR+openfda.generic_name:{medicine_name}&limit=1&api_key={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    uses = []
                    
                    if 'indications_and_usage' in result:
                        content = result['indications_and_usage']
                        if isinstance(content, list):
                            uses.extend(content)
                        elif isinstance(content, str):
                            uses.append(content)
                    
                    cleaned = []
                    for use in uses:
                        use = use.strip()
                        if use and len(use) > 5:
                            cleaned.append(use)
                    
                    return cleaned[:5]
        except Exception as e:
            print(f"⚠️ FDA API error: {e}")
        
        return []


class MedicinePredictor:
    """Combines ML predictions and FDA data"""
    
    def __init__(self):
        self.ml = MLPredictor()
        self.fda = FDAApi()
    
    def get_predictions(self, medicine_name: str, contains: List[str] = None) -> List[str]:
        """
        Get medicine use predictions from both ML and FDA sources
        """
        all_uses = []
        
        # Get ML predictions
        ml_uses = self.ml.predict_uses(medicine_name, contains)
        if ml_uses and "Error" not in ml_uses[0]:
            all_uses.extend(ml_uses)
        
        # Get FDA data
        fda_uses = self.fda.get_medicine_uses(medicine_name)
        if fda_uses:
            all_uses.extend(fda_uses)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_uses))