# entity_extraction.py
# Handles extraction of medicine name, chemicals, composition, and ingredients

import re
import pandas as pd
import spacy
import streamlit as st
from utils import text_cleaner  # Import from existing utils


class EntityExtractor:
    """Extracts medicine entities from text"""
    
    def __init__(self):
        self.nlp = self._load_spacy()
        self.dosage_forms = [
            "tablet", "tablets", "capsule", "capsules",
            "syrup", "drop", "drops", "injection",
            "cream", "gel", "ointment", "respules",
            "suspension", "solution"
        ]
    
    @staticmethod
    def _load_spacy():
        """Load spaCy model with caching"""
        try:
            nlp = spacy.load("en_ner_bc5cdr_md")
            print("✅ spaCy biomedical model loaded")
            return nlp
        except:
            try:
                nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy (fallback) loaded")
                return nlp
            except:
                print("❌ spaCy not available")
                return None
    
    def _ensure_string(self, text):
        """Helper method to ensure text is a clean string"""
        if text is None:
            return ""
        
        # Handle pandas Series
        if isinstance(text, pd.Series):
            # Convert Series to string, handling NaN values
            text = " ".join(text.dropna().astype(str))
        
        # Handle lists or tuples
        if isinstance(text, (list, tuple)):
            text = " ".join([str(x) for x in text if x is not None])
        
        # Ensure it's a string
        text = str(text).strip()
        
        # Remove any pandas-specific artifacts
        if text.startswith("0    ") or text.startswith("dtype:"):
            # This might be a pandas Series string representation
            lines = text.split('\n')
            # Take only the actual content lines
            text = " ".join([line for line in lines if not line.strip().startswith('dtype:')])
        
        return text.strip()
    
    def extract_chemicals(self, text):
        """Extract chemical names using spaCy NER"""
        if self.nlp is None:
            return []
        
        text = self._ensure_string(text)
        if text == "":
            return []
        
        try:
            doc = self.nlp(text)
            chemicals = []
            for ent in doc.ents:
                if ent.label_ == "CHEMICAL":
                    chemicals.append(ent.text.strip())
            return sorted(list(set(chemicals)))
        except Exception as e:
            print(f"⚠️ Chemical extraction error: {e}")
            return []
    
    def extract_composition(self, text):
        """Extract composition using regex"""
        text = self._ensure_string(text)
        if text == "":
            return []
        
        pattern = re.compile(
            r'([A-Za-z][A-Za-z0-9()\-\/\s]{2,60}\s\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|%)'
            r'(?:\/\d+\s*(?:ml|g))?)',
            re.IGNORECASE
        )
        
        try:
            result = []
            for match in pattern.findall(text):
                result.append(" ".join(match.split()))
            return list(dict.fromkeys(result))
        except Exception as e:
            print(f"⚠️ Composition extraction error: {e}")
            return []
    
    def extract_medicine_name(self, text):
        """Extract medicine name using dosage form patterns"""
        text = self._ensure_string(text)
        if text == "":
            return None
        
        # Apply text cleaner - ensure it returns a string
        try:
            cleaned = text_cleaner(text)
            if cleaned is not None:
                text = str(cleaned).strip()
            if not text:
                return None
        except Exception as e:
            print(f"⚠️ Text cleaner error: {e}")
            # Continue with original text
        
        for form in self.dosage_forms:
            pattern = rf"([A-Z][A-Za-z0-9\- ]{{2,40}})\s+{form}"
            try:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    name = m.group(1)
                    name = re.sub(r"\b(IP|USP|BP)\b", "", name, flags=re.I)
                    return name.strip()
            except Exception as e:
                print(f"⚠️ Medicine name extraction error: {e}")
                continue
        
        return None
    
    def extract_ingredients(self, text):
        """Extract ingredients using regex"""
        text = self._ensure_string(text)
        if text == "":
            return []
        
        ingredients = []
        ingredient_patterns = [
            r'contains?\s*[:]\s*([^.]*)',
            r'active\s*ingredient[s]?\s*[:]\s*([^.]*)',
            r'composition[s]?\s*[:]\s*([^.]*)',
            r'each\s*(?:tablet|capsule)\s*contains?\s*([^.]*)'
        ]
        
        try:
            for pattern in ingredient_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        parts = re.split(r'[,;]', match)
                        ingredients.extend([p.strip() for p in parts if p.strip()])
            return ingredients
        except Exception as e:
            print(f"⚠️ Ingredients extraction error: {e}")
            return []
    
    def extract_all_info(self, text):
        """Extract all medicine information"""
        # First ensure text is a clean string
        text = self._ensure_string(text)
        
        if text == "":
            return {
                'chemicals': [],
                'composition': [],
                'medicine_name': None,
                'ingredients': [],
                'clean_text': text
            }
        
        # Clean the text - handle potential Series return
        try:
            clean = text_cleaner(text)
            if clean is not None:
                if isinstance(clean, pd.Series):
                    clean = " ".join(clean.dropna().astype(str))
                clean = str(clean).strip()
            else:
                clean = text
        except Exception as e:
            print(f"⚠️ Text cleaning error: {e}")
            clean = text
        
        # Extract all information
        chemicals = self.extract_chemicals(clean)
        composition = self.extract_composition(clean)
        medicine_name = self.extract_medicine_name(clean)
        ingredients = self.extract_ingredients(clean)
        
        return {
            'chemicals': chemicals,
            'composition': composition,
            'medicine_name': medicine_name,
            'ingredients': ingredients,
            'clean_text': clean
        }