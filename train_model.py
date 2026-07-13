# train_model.py - Optimized for Large Dataset
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import warnings
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import Counter
import os

warnings.filterwarnings('ignore')

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)

# ============================================
# TEXT CLEANER FUNCTION
# ============================================

lemmatizer = WordNetLemmatizer()

def get_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token, get_pos(token)) for token in tokens]
    return " ".join(lemmas)

def text_cleaner(x):
    return x.apply(clean_text)

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("📂 Loading data...")

# Check if Excel file exists
if not os.path.exists('MID.xlsx'):
    print("❌ MID.xlsx not found!")
    print("Please make sure the file exists in the current directory.")
    exit()

data1 = pd.read_excel('MID.xlsx')

# Select relevant columns
data2 = data1[['NAME', 'CONTAINS', 'USES', 'SIDE_EFFECT', 'HOW_TO_USE', 
               'HABIT_FORMING', 'SAFETY_ADVICE_TO_ALCOHOL', 
               'SAFETY_ADVICE_TO_PREGNANCY', 'SAFETY_ADVICE_TO_BREAST_FEEDING', 
               'SAFETY_ADVICE_TO_DRIVING', 'SAFETY_ADVICE_TO_KIDNEY', 
               'SAFETY_ADVICE_TO_LIVER']]

print(f"✅ Loaded {len(data2)} records")

# Clean USES column
data2['USES'] = data2['USES'].fillna('')
data2['USES'] = data2['USES'].apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])

# ============================================
# ⭐ OPTIMIZATION 1: Limit number of target labels
# ============================================

print("🔄 Analyzing target labels...")

# Get all unique uses
all_uses = []
for uses in data2['USES']:
    all_uses.extend(uses)

# Count frequency of each use
use_counts = Counter(all_uses)
print(f"Total unique uses: {len(use_counts)}")

# Keep only top N most frequent uses (reduce memory)
TOP_N_USES = 50  # Adjust this number based on your needs
top_uses = [use for use, count in use_counts.most_common(TOP_N_USES)]
print(f"Keeping top {TOP_N_USES} most frequent uses")

# Filter data to only keep top uses
def filter_uses(uses_list):
    return [use for use in uses_list if use in top_uses]

data2['USES_filtered'] = data2['USES'].apply(filter_uses)

# Remove rows with no filtered uses
data2 = data2[data2['USES_filtered'].apply(len) > 0]
print(f"✅ After filtering: {len(data2)} records")

# ============================================
# CREATE INPUT
# ============================================

print("🔄 Creating input text...")

data2['NAME'] = data2['NAME'].fillna('')
data2['CONTAINS'] = data2['CONTAINS'].fillna('')

data2['input'] = (
    data2['NAME'].astype(str) + " " +
    data2['CONTAINS'].astype(str)
)

# ============================================
# SAMPLE DATA (DO THIS BEFORE CREATING y)
# ============================================

print("🔄 Sampling data...")

MAX_SAMPLES = 50000

if len(data2) > MAX_SAMPLES:
    data2 = (
        data2
        .sample(n=MAX_SAMPLES, random_state=42)
        .reset_index(drop=True)
    )
    print(f"✅ Sampled to {len(data2)} records")
else:
    data2 = data2.reset_index(drop=True)

# ============================================
# PREPARE TARGET (AFTER SAMPLING)
# ============================================

print("🔄 Preparing target labels...")

mlb = MultiLabelBinarizer(classes=top_uses)

y = mlb.fit_transform(data2['USES_filtered'])

X = data2['input']

print(f"✅ X shape : {X.shape}")
print(f"✅ y shape : {y.shape}")

assert len(X) == len(y), \
    f"Mismatch! X={len(X)}, y={len(y)}"

print(f"✅ Target labels: {len(mlb.classes_)}")

# ============================================
# CREATE PIPELINE
# ============================================

print("🔄 Creating pipeline...")

pipe = Pipeline([
    ('clean', FunctionTransformer(text_cleaner)),
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=500
    )),
    ('model', MultiOutputClassifier(
        LogisticRegression(
            max_iter=500,
            random_state=42,
            C=1.0
        ),
        n_jobs=1
    ))
])

# ============================================
# TRAIN MODEL
# ============================================

print("🔄 Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"Train samples : {len(X_train)}")
print(f"Test samples  : {len(X_test)}")

print("⏳ Training model...")

pipe.fit(X_train, y_train)

print("✅ Model trained successfully!")
# ============================================
# TEST PREDICTION
# ============================================

print("\n" + "="*50)
print("🧪 Testing prediction...")
print("="*50)

test_medicines = ["Paracetamol", "Ibuprofen", "Amoxicillin"]

for med in test_medicines:
    try:
        # Get the first match
        med_data = data2[data2['NAME'].str.contains(med, case=False, na=False)]
        if not med_data.empty:
            contains = med_data.iloc[0]['CONTAINS'] if pd.notna(med_data.iloc[0]['CONTAINS']) else ""
            input_text = f"{med} {contains}"
            
            pred = pipe.predict([input_text])
            pred_labels = mlb.inverse_transform(pred)
            
            print(f"\n💊 {med}")
            print(f"   Input: {input_text[:50]}...")
            if pred_labels[0]:
                print(f"   Uses: {', '.join(list(pred_labels[0])[:5])}")
            else:
                print("   No uses predicted")
        else:
            print(f"\n💊 {med}: Not found in dataset")
    except Exception as e:
        print(f"\n💊 {med}: Error - {e}")

print("\n✅ Training complete!")