import joblib
import numpy as np

pipe = joblib.load("medicine_model.pkl")
mlb = joblib.load("label_encoder.pkl")

tests = [
    "Paracetamol Paracetamol 500 mg",
    "Ibuprofen Ibuprofen 400 mg",
    "Omeprazole Omeprazole 20 mg",
    "Tramadol Hydrochloride Acetaminophen 325 mg Tramadol Hydrochloride 37.5 mg"
]

for text in tests:
    print("=" * 80)
    print("INPUT:", text)

    pred = pipe.predict([text])
    labels = mlb.inverse_transform(pred)

    print("Prediction matrix sum:", pred.sum())
    print("Predicted labels:", labels)

    probs = pipe.predict_proba([text])[0]
    top = np.argsort(probs)[-5:][::-1]

    print("\nTop 5 probabilities:")
    for i in top:
        print(f"{mlb.classes_[i]:50} {probs[i]:.4f}")