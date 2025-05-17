import streamlit as st
from ingestion import fetch_transcript_from_video, transcript_to_text
from embedding_store import create_embeddings
from retrieval_chain import build_qa_chain_gemini
import time
import re
from fpdf import FPDF
from textblob import TextBlob  # Sentiment analysis

st.set_page_config(page_title="Gemini YouTube Q&A", layout="wide")
st.title("🎥 Chat with YouTube Video using Gemini")

# --- Session State Initialization ---
for key in ["vector_store", "chat_chain", "chat_history", "load_clicked", "edited_transcript", "video_id"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# --- Helper Functions ---
def extract_video_id(input_str: str) -> str:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, input_str)
    if match:
        return match.group(1)
    return None

def is_valid_youtube_url(url: str) -> bool:
    pattern = r"^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$"
    return re.match(pattern, url.strip()) is not None

def save_text_file(text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def save_pdf_file(text: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    lines = [text[i:i+90] for i in range(0, len(text), 90)]
    for line in lines:
        pdf.cell(0, 10, line, ln=True)
    pdf.output(filename)

def analyze_sentiment(text: str):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.3:
        tone = "Positive"
    elif polarity < -0.3:
        tone = "Negative"
    else:
        tone = "Neutral"

    return tone, polarity, subjectivity

# --- UI ---
st.subheader("📥 Enter YouTube Video URL or Video ID")
video_input = st.text_input("Paste YouTube URL or enter Video ID", placeholder="Youtube URL or Video ID")

if st.button("🚀 Load Transcript and Initialize Chat"):
    if not video_input.strip():
        st.error("Please enter a valid YouTube video URL or ID.")
        st.stop()

    # Validate YouTube URL/ID
    if not is_valid_youtube_url(video_input):
        st.error("Invalid YouTube URL.")
        st.stop()

    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("Could not extract a valid YouTube video ID.")
        st.stop()
    st.session_state.video_id = video_id
    st.session_state.load_clicked = True

if st.session_state.load_clicked:
    st.write(f"🔍 Extracted Video ID: `{st.session_state.video_id}`")
    status = st.empty()
    status.info("📡 Fetching transcript...")
    time.sleep(0.5)

    transcript_segments = fetch_transcript_from_video(video_input)
    if not transcript_segments:
        status.error("❌ Transcript not found or disabled.")
        st.session_state.load_clicked = False
        st.stop()

    transcript_text = transcript_to_text(transcript_segments)
    st.session_state.edited_transcript = transcript_text

    status.success("✅ Transcript fetched successfully.")
    time.sleep(1)
    status.empty()

    # --- Transcript Downloads ---
    st.subheader("⬇️ Download Transcript")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download as TXT"):
            save_text_file(transcript_text, "transcript.txt")
            st.success("Transcript saved as transcript.txt")
    with col2:
        if st.button("Download as PDF"):
            save_pdf_file(transcript_text, "transcript.pdf")
            st.success("Transcript saved as transcript.pdf")

    # --- Sentiment & Tone Analysis ---
    st.subheader("🎭 Sentiment & Tone Analysis")
    tone, polarity, subjectivity = analyze_sentiment(transcript_text)
    st.markdown(f"""
    - **Overall Tone:** `{tone}`
    - **Polarity Score:** `{polarity:.2f}`  
    - **Subjectivity Score:** `{subjectivity:.2f}`
    """)
    st.info("Polarity ranges from -1 (Negative) to +1 (Positive). Subjectivity ranges from 0 (Objective) to 1 (Subjective).")

    # --- Create Vector Store ---
    status.info("🧠 Creating embeddings & building index...")
    time.sleep(0.5)
    vector_store = create_embeddings(transcript_text)
    if not vector_store:
        status.error("❌ Failed to build vector store.")
        st.session_state.load_clicked = False
        st.stop()

    st.session_state.vector_store = vector_store
    status.success("✅ Vector store created.")
    time.sleep(1)
    status.empty()

    # --- Initialize QA Chain ---
    try:
        st.session_state.chat_chain = build_qa_chain_gemini(vector_store)
        st.session_state.chat_history = []
        st.success("🤖 Chatbot is ready to answer your questions!")
    except Exception as e:
        st.error(f"❌ Error initializing chat chain: {e}")
        st.session_state.load_clicked = False
        st.stop()

    st.session_state.load_clicked = False

# --- Chat Q&A Interface ---
if st.session_state.chat_chain:
    st.subheader("💬 Q&A History")

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # Sticky container for answer style selector just above input
    style_container = st.container()
    with style_container:
        answer_style = st.selectbox(
            "🛠️ Choose Answer Style",
            options=["Default", "Answer like a teacher", "Answer with bullet points", "Summarize the answer", "Add examples"],
            index=0
        )

    user_input = st.chat_input("Ask your question…")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Style modifier
        style_prompt = {
            "Answer like a teacher": "Explain the answer like a teacher would to a student.",
            "Answer with bullet points": "Answer using concise bullet points.",
            "Summarize the answer": "Provide a short summary of the answer.",
            "Add examples": "Add relevant examples in the answer.",
            "Default": ""
        }.get(answer_style, "")

        full_prompt = f"{style_prompt} {user_input}".strip()

        with st.chat_message("assistant"):
            try:
                with st.spinner("Gemini is thinking…"):
                    reply = st.session_state.chat_chain.invoke(full_prompt)

                    # Optional cleanup
                    if reply.startswith("Based on the provided transcript context,"):
                        reply = reply.replace("Based on the provided transcript context,", "").strip()

                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error generating answer: {e}")
