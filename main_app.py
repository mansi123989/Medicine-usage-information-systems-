# main_app.py
# Main Streamlit application - coordinates all modules

import os


# Environment settings - MUST BE FIRST
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import base64
import tempfile
import hashlib
import streamlit as st
from PIL import Image
from gtts import gTTS
from dotenv import load_dotenv
from typing import List
import re

# Import modules
from text_extraction_correction import TextExtractor, TextCorrector, get_groq_client
from entity_extraction import EntityExtractor
from medicine_predictor import MedicinePredictor, Config
from utils import text_cleaner

# Environment settings - MUST BE FIRST
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load environment variables
load_dotenv()

# ⭐ MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="💊 Medicine Information Assistant",
    page_icon="💊",
    layout="wide"
)

# ============================================
# LANGUAGE SUPPORT
# ============================================

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Korean": "ko",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Urdu": "ur"
}

# ============================================
# GROQ EXPLANATION FUNCTION
# ============================================

def get_simple_medicine_explanation(groq_client, medicine_name: str, uses: List[str], target_language: str = "English") -> str:
    """
    Use Groq to create a simple, general explanation of what the medicine does
    Returns ONE simple sentence explaining the medicine's purpose in everyday language
    """
    if not groq_client or not uses:
        return "This medicine helps with health problems."
    
    # Prepare the prompt for Groq
    uses_text = "\n".join([f"{i+1}. {use}" for i, use in enumerate(uses[:8])])
    
    prompt = f"""
You are a helpful medical assistant that explains medicines in very simple, everyday language for illiterate people.

Medicine: {medicine_name}

What this medicine is used for (medical terms):
{uses_text}

Please create ONE SIMPLE, SHORT, AND EASY-TO-UNDERSTAND explanation of what this medicine does.

IMPORTANT RULES:
1. Start with: "This medicine is used for" or "This medicine helps with"
2. DO NOT mention the medicine name again - just say "this medicine"
3. Use very simple words that anyone can understand
4. Keep it to 1-2 sentences maximum
5. No medical jargon at all
6. Make it general and easy to remember
7. Think about how you would explain this to a child or someone who cannot read

Example good responses:
- "This medicine helps reduce fever and body pain"
- "This medicine is used to treat stomach infections"
- "This medicine helps control high blood pressure"
- "This medicine is for treating skin allergies and rashes"

The response MUST be in {target_language} language.

ONLY return the simple explanation text, nothing else.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You explain medicines in very simple terms for people who cannot read or understand medical language."},
                {"role": "user", "content": prompt}
            ]
        )
        
        explanation = response.choices[0].message.content.strip()
        
        # Clean up the response
        explanation = re.sub(r'^["\']|["\']$', '', explanation)  # Remove quotes
        explanation = explanation.replace('"', '').replace("'", "")
        
        # Ensure it starts with appropriate phrase if not already
        if not explanation.lower().startswith(('this medicine', 'it is used', 'used for', 'helps with')):
            explanation = "This medicine " + explanation.lower()
        
        return explanation
            
    except Exception as e:
        print(f"❌ Groq explanation error: {e}")
        # Fallback to simple generated explanation
        if uses:
            first_use = uses[0].lower()
            first_use = re.sub(r'^(indicated for|used for|treatment of|for the treatment of|for)', '', first_use)
            return f"This medicine helps with {first_use.strip()}"
        return "This medicine helps with health problems."

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Apply custom CSS
    st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #2c3e50; padding: 1rem 0; }
        .stButton > button { width: 100%; background-color: #3498db; color: white; font-weight: bold; }
        .stButton > button:hover { background-color: #2980b9; }
        .success-box { background: #d4edda; padding: 1rem; border-radius: 5px; border-left: 4px solid #28a745; }
        .ml-badge { background: #27ae60; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; }
        .fda-badge { background: #2c3e50; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; }
        .ner-badge { background: #6c5ce7; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; }
        .warning-box { background: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 4px solid #ffc107; }
        .explanation-box {
            background: #e8f4fd;
            padding: 25px;
            border-radius: 12px;
            border-left: 6px solid #2196F3;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .explanation-text {
            font-size: 1.4rem;
            font-weight: 500;
            color: #1a237e;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">💊 Medicine Information Assistant</h1>', unsafe_allow_html=True)
    st.markdown("📸 Upload a photo of your medicine packaging to know what it does")
    
    # Initialize components
    text_extractor = TextExtractor()
    entity_extractor = EntityExtractor()
    predictor = MedicinePredictor()
    groq_client = get_groq_client(os.getenv("GROQ_API_KEY"))
    text_corrector = TextCorrector(groq_client) if groq_client else None
    
    # Language selection in sidebar
    with st.sidebar:
        st.header("🌍 Language Settings")
        target_language = st.selectbox(
            "Select your preferred language:",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )
        st.info("💡 The medicine information will be explained in simple words in your language.")
    
    # Image upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        camera_image = st.camera_input("📸 Click Image")
        uploaded = st.file_uploader(
            "📁 Or upload from gallery...",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Take a clear photo of the medicine box or strip or upload an existing image."
        )
    
    uploaded_file = camera_image if camera_image is not None else uploaded
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Hash-based cache clearing
        image_bytes = uploaded_file.getvalue()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        if st.session_state.get("image_hash") != image_hash:
            st.session_state.clear()
            st.session_state["image_hash"] = image_hash
        
        # Display image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Medicine Package")
        
        with st.spinner("🔍 Analyzing and simplifying information..."):
            
            # Step 1: Extract text with EasyOCR
            if "raw_text" not in st.session_state:
                st.session_state["raw_text"] = text_extractor.extract_text(image)
            raw_text = st.session_state["raw_text"]
            
            if raw_text:
                # Step 2: Correct text with Groq
                if text_corrector:
                    if "corrected_text" not in st.session_state:
                        st.session_state["corrected_text"] = text_corrector.correct_text_with_groq(raw_text)
                    corrected_text = st.session_state["corrected_text"]
                else:
                    corrected_text = raw_text
                    st.warning("⚠️ Using basic OCR without correction")
                
                # Step 3: Extract all information
                if "extraction_results" not in st.session_state:
                    st.session_state["extraction_results"] = entity_extractor.extract_all_info(corrected_text)
                extraction_results = st.session_state["extraction_results"]
                
                medicine_name = extraction_results["medicine_name"]
                
                if medicine_name:
                    # Prepare ingredients for ML model
                    contains = []
                    contains.extend(extraction_results["composition"])
                    contains.extend(extraction_results["chemicals"])
                    contains.extend(extraction_results["ingredients"])
                    contains = list(dict.fromkeys([x.strip() for x in contains if x.strip()]))
                    
                    # Step 4: Get predictions (ML + FDA)
                    if "all_uses" not in st.session_state:
                        st.session_state["all_uses"] = predictor.get_predictions(medicine_name, contains)
                    all_uses = st.session_state["all_uses"]
                    
                    if all_uses:
                        # Step 5: Get simple explanation with Groq
                        if groq_client:
                            simple_explanation = get_simple_medicine_explanation(
                                groq_client,
                                medicine_name,
                                all_uses,
                                target_language
                            )
                        else:
                            # Fallback explanation without Groq
                            first_use = all_uses[0].lower()
                            first_use = re.sub(r'^(indicated for|used for|treatment of|for the treatment of|for)', '', first_use)
                            simple_explanation = f"This medicine helps with {first_use.strip()}"
                            st.warning("⚠️ Groq API not available. Using fallback explanation.")
                        
                        # Store in session state
                        st.session_state["medicine_name"] = medicine_name
                        st.session_state["simple_explanation"] = simple_explanation
                        st.session_state["all_uses"] = all_uses
                        st.session_state["target_language"] = target_language
                        st.session_state["speech_text"] = simple_explanation
                        
                        # Display the explanation
                        with col2:
                            st.markdown("### 🎯 What this medicine does")
                            st.markdown(f"""
                            <div class="explanation-box">
                                <div class="explanation-text">
                                    {simple_explanation}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption(f"🌍 Information in {target_language}")
                            st.success("✅ Audio ready - Click the button below to listen")
                    else:
                        st.error("❌ Could not determine what this medicine does")
                        st.info("💡 Try taking a clearer photo of the medicine name")
                else:
                    st.error("❌ Could not read the medicine name")
                    st.info("💡 Tips: Use clear photos, ensure medicine name is visible and readable")
            else:
                st.error("❌ No text could be read from the image")
                st.info("💡 Tips: Use good lighting, clear focus, and make sure the text is not blurry")
        
        # Audio playback section
        if "simple_explanation" in st.session_state and "speech_text" in st.session_state:
            st.markdown("---")
            st.markdown("### 🔊 Listen to the explanation")
            
            try:
                language_code = SUPPORTED_LANGUAGES.get(
                    st.session_state.get("target_language", "English"),
                    "en"
                )
                tts = gTTS(
                    text=st.session_state["speech_text"],
                    lang=language_code,
                    slow=False
                )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tts.save(tmp.name)
                    with open(tmp.name, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                
                # Auto-play on load
                st.markdown(
                    f"""
                    <audio autoplay id="medicine_audio">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Big listen button
                if st.button("🔊 PLAY AUDIO", use_container_width=True):
                    st.markdown(
                        f"""
                        <audio autoplay controls style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        </audio>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.info("🎵 Audio is playing... Click again if you need to replay")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate audio: {e}")
                st.info("Please try again or check your internet connection")
        
        # Technical details expander
        if "all_uses" in st.session_state and st.session_state["all_uses"]:
            with st.expander("📋 Technical information (for reference)"):
                st.write("Medicine Name:", st.session_state.get("medicine_name", "Unknown"))
                st.write("Medical uses identified:")
                for use in st.session_state["all_uses"][:5]:
                    st.write(f"• {use}")

if __name__ == "__main__":
    main()