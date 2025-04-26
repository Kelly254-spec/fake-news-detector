import streamlit as st
from textblob import TextBlob
import speech_recognition as sr
import pickle
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title with emoji
st.title("🕵️‍♂️ Fake News Detector")
st.write("Upload a file or paste an article to check if it's **Fake** or **Real**.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to fetch article content from URL (basic)
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except:
        return "Unable to extract content from the provided URL."

# File uploader
st.subheader("📎 Upload a File (Text or PDF)")
uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

# URL section
st.subheader("🔗 Or Paste a URL")
url_input = st.text_input("Enter URL here")

# Manual input section
st.subheader("📝 Or Paste Article Text")
manual_text = st.text_area("Paste the article text here")

# Voice input section
st.subheader("🎤 Speak to Check")

recognizer = sr.Recognizer()

# Button to activate voice input with a unique key
if st.button("🎙️ Start Speaking", key="start_speaking_button_1"):
    with sr.Microphone() as source:
        st.write("🎤 Listening... Please speak now.")
        audio = recognizer.listen(source)
        try:
            speech_text = recognizer.recognize_google(audio)  # Convert speech to text
            st.write(f"📢 You said: {speech_text}")
            
            # Predict based on the speech
            vectorized = vectorizer.transform([speech_text])  # Transform speech to vector
            prediction = model.predict(vectorized)[0]
            
            # Display result
            if prediction == "FAKE":
                st.error("⚠️ This article is likely **FAKE**.")
            else:
                st.success("✅ This article is likely **REAL**.")
        except sr.UnknownValueError:
            st.error("❌ Sorry, I couldn't understand your speech.")
        except sr.RequestError:
            st.error("❌ Sorry, there was an issue with the speech recognition service.")

# Language Detection and Translation Section
st.subheader("🌍 Language Detection and Translation")
language_text = st.text_area("📝 Paste text for language detection and translation")

if language_text:
    blob = TextBlob(language_text)
    
    # Detect language
    detected_language = blob.detect_language()

    # Display detected language
    st.write(f"Detected Language: {detected_language}")
    
    # Translate to English if the text is not already in English
    if detected_language != "en":
        translated_text = blob.translate(to="en")
        st.write("Translated Text:")
        st.write(translated_text)
    else:
        st.write("Text is already in English.")

# Determine content source
article = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        article = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        article = uploaded_file.read().decode('utf-8')

elif url_input:
    article = extract_text_from_url(url_input)

elif manual_text:
    article = manual_text

# Analyze the article
if article:
    st.subheader("🧠 Analysis Result:")
    vectorized = vectorizer.transform([article])
    prediction = model.predict(vectorized)[0]
    if prediction == "FAKE":
        st.error("⚠️ This article is likely **FAKE**.")
    else:
        st.success("✅ This article is likely **REAL**.")

# Trending Fakes Section
st.markdown("---")
st.subheader("🔥 Trending Fakes Today")

# Dummy trending fake headlines (we'll connect real-time later)
trending_fakes = [
    "🚨 'Aliens Landed in Lagos!' - Viral Twitter Post",
    "💉 'New Vaccine Turns People into Zombies' - WhatsApp Forward",
    "📉 'Stock Market Crash Predicted by Pastor!' - FakeNews247",
    "💰 'You Can Get Rich from One App' - Viral TikTok Claim",
    "👑 'Queen of England Adopts a Baby Tiger' - Facebook Article",
]

for news in trending_fakes:
    st.write(f"- {news}")


