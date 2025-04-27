import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator
from newspaper import Article
import pickle

# Load your model
model = pickle.load(open("model.pkl", "rb"))

# Function to detect language and translate
def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != 'en':
            with st.spinner(f"Detected language '{lang}'. Translating to English..."):
                translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            return translated_text
        else:
            return text
    except Exception as e:
        st.error(f"Language detection or translation failed: {e}")
        return text

# Streamlit App
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Detect whether a news article is **Real** or **Fake** easily! 🔍")

# Input selection
input_type = st.selectbox("Choose Input Type:", ["Upload a File", "URL", "Article Text"])

# Main app logic
if input_type == "Upload a File":
    st.subheader("📄 Upload a News File")
    uploaded_file = st.file_uploader("Choose a text file (.txt)")
    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')
        text = detect_and_translate(text)
        with st.spinner("Analyzing the news..."):
            prediction = model.predict([text])
        st.success(f"✅ Prediction: **{'Fake' if prediction[0] == 1 else 'Real'}**")

elif input_type == "URL":
    st.subheader("🔗 Enter a News Article URL")
    user_input = st.text_input("Paste the URL here:")
    if user_input:
        try:
            with st.spinner("Downloading and analyzing the article..."):
                article = Article(user_input)
                article.download()
                article.parse()
                text = article.text
                text = detect_and_translate(text)
                prediction = model.predict([text])
            st.success(f"✅ Prediction: **{'Fake' if prediction[0] == 1 else 'Real'}**")
        except Exception as e:
            st.error(f"Failed to process the URL. Please check the link. Error: {e}")

elif input_type == "Article Text":
    st.subheader("✍️ Paste News Article Text")
    user_input = st.text_area("Paste the full article text here:")
    if user_input:
        text = detect_and_translate(user_input)
        with st.spinner("Analyzing the news..."):
            prediction = model.predict([text])
        st.success(f"✅ Prediction: **{'Fake' if prediction[0] == 1 else 'Real'}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning and Streamlit.")


