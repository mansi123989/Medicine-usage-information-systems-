# text_extraction_correction.py
# Handles OCR text extraction and correction using EasyOCR and Groq

import os
import re
import numpy as np
from PIL import Image
import easyocr
import streamlit as st
from groq import Groq
from utils import text_cleaner  # Import from existing utils

# Environment settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


class TextExtractor:
    """Handles OCR text extraction from images"""
    
    def __init__(self):
        self.reader = self._load_easyocr()
    
    @staticmethod
    def _load_easyocr():
        """Load EasyOCR reader with caching"""
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR loaded")
            return reader
        except Exception as e:
            print(f"❌ EasyOCR failed: {e}")
            return None
    
    def extract_text(self, image):
        """Extract text using EasyOCR"""
        if self.reader is None:
            return ""
        
        try:
            image_np = np.array(image)
            result = self.reader.readtext(
                image_np,
                detail=0,
                paragraph=True
            )
            return "\n".join(result)
        except Exception as e:
            print(f"❌ OCR extraction error: {e}")
            return ""


class TextCorrector:
    """Handles OCR text correction using Groq LLM"""
    
    def __init__(self, groq_client):
        self.groq_client = groq_client
    
    def correct_text_with_groq(self, text):
        """Correct OCR text using Groq LLM"""
        if not self.groq_client or not text:
            return text
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        if not text.strip():
            return text
        
        prompt = f"""
You are an OCR correction assistant.

The following text was extracted from a medicine package.

Correct spelling mistakes caused by OCR.

Rules:
1. Preserve medicine names exactly.
2. Preserve chemical compositions.
3. Do not add information.
4. Do not summarize.
5. Return only corrected text.

OCR Text:

{text}
"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Groq correction error: {e}")
            return text


def get_groq_client(api_key):
    """Initialize Groq client"""
    if not api_key:
        print("⚠️ GROQ_API_KEY not found")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"❌ Groq client error: {e}")
        return None