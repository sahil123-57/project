# resume_screening_app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

app = FastAPI()

# Enable CORS (allowing cross-origin requests for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "ml_model.pkl"


# ========================== #
# 📌 PDF Text Extraction Logic
# ========================== #
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ========================== #
# 📌 Load or Train ML Model
# ========================== #
def train_and_save_model():
    resume_texts = [
        "Python developer with machine learning experience",
        "Sales manager with strong communication skills",
        "Experienced Java backend developer",
        "Data scientist with strong Python and statistics skills",
        "Digital marketing specialist with SEO knowledge",
    ]
    labels = [1, 0, 0, 1, 0]  # 1 = suitable, 0 = unsuitable

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(resume_texts)

    model = RandomForestClassifier()
    model.fit(X, labels)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, tfidf), f)
    print(f"✅ Model trained and saved to {MODEL_FILE}")


def load_model():
    if not os.path.exists(MODEL_FILE):
        train_and_save_model()
    with open(MODEL_FILE, "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_model()


# ========================== #
# 📌 Prediction Logic
# ========================== #
def predict_resume_score(resume_text):
    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)[0]
    return int(prediction)


# ========================== #
# 📌 FastAPI Endpoint
# ========================== #
@app.post("/screen_resume/")
async def screen_resume(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    extracted_text = extract_text_from_pdf(pdf_bytes)
    suitability_score = predict_resume_score(extracted_text)

    return {
        "resume_text_preview": extracted_text[:500],  # Preview of the resume text
        "suitability_score": suitability_score
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("resume_screening_app:app", host="127.0.0.1", port=8000, reload=True)
