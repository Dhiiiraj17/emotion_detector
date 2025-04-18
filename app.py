import streamlit as st
from transformers import pipeline
import torch
import nltk
nltk.download("punkt")

# Set device
device = 0 if torch.cuda.is_available() else -1

# Emotion detection pipeline
emotion_detector = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False,
    device=device
)

# Bot response templates
responses = {
    "joy": "😊 That's awesome! I'm so glad you're feeling happy today! Keep that positivity flowing!",
    "sadness": "💙 It's perfectly okay to feel down sometimes. I'm here to listen. Want to talk more about it?",
    "anger": "😡 I can feel that frustration. Sometimes venting helps! Want to share what's bothering you?",
    "fear": "😨 It's totally okay to feel scared. You're not alone, I'm right here with you. Let's talk about it.",
    "surprise": "😲 Wow, that’s an unexpected twist! Life sure knows how to keep us on our toes, huh?",
    "love": "❤️ Aww, that’s so sweet! Love and kindness are powerful. Let's cherish those moments!"
}

# Streamlit page config
st.set_page_config(page_title="Emotion Chatbot 🤖", layout="centered")
st.markdown("<h1 style='text-align:center;'>🧠 Emotion-Aware Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Tell me how you're feeling 💬</p>", unsafe_allow_html=True)

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# User input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your feelings...")
    submit = st.form_submit_button("Send")

# Handle new message
if submit and user_input.strip():
    # Get emotion
    result = emotion_detector(user_input)[0]
    emotion = result["label"].lower()
    score = round(result["score"], 2)
    bot_response = responses.get(emotion, "🤖 I'm here for you. Let's talk!")

    # Save to chat history
    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", f"{bot_response} *(Detected: {emotion} - {score})*"))

# Display chat
for sender, message in st.session_state.chat:
    if sender == "user":
        st.markdown(
            f"""
            <div style='text-align: right; background-color: #1A73E8; padding: 12px 18px; 
                        border-radius: 15px; margin: 12px 0 12px auto; max-width: 80%;'>
                <strong style='color: white;'>You</strong>: {message}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style='text-align: left; background-color: #4CAF50; padding: 12px 18px; 
                        border-radius: 15px; margin: 12px auto 12px 0; max-width: 80%;'>
                <strong style='color: white;'>Bot</strong>: {message}
            </div>
            """, unsafe_allow_html=True)
