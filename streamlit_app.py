import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
nltk.download('stopwords')

def clean_text(text):

    text=''.join([char for char in text if char not in string.punctuation])
    words=text.split()
    stop_words=stopwords.words("english")
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)


# Load pre-trained objects
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit app settings
st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detection Web App")
st.write("Use this app to check if an email or message is **Spam** or **Not Spam** using your trained model.")

# Text input area for user email/message
user_input = st.text_area("✉️ Enter email or message text:", height=200)

# When user clicks "Check Spam"
if st.button("Check Spam"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    elif len(user_input.split()) < 10:
        st.warning("⚠️ Please enter at least 10 words for better prediction accuracy.")
    else:
        try:
            # Use your existing preprocessing and model
            processed_text =clean_text(user_input)
            vector_input = vectorizer.transform([processed_text])
            prediction = model.predict(vector_input)

            # Show result
            if prediction > 0.7:
                st.error("🚨 This message is classified as **SPAM**.")
            else:
                st.success("✅ This message is classified as **NOT SPAM**.")

        except Exception as e:
            st.warning(f"⚠️ Error during prediction: {e}")

st.markdown("---")
st.caption("Trained model from your existing code is used for prediction.")
st.caption("BY RUCHIT JAIN GITHUB")













