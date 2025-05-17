# 🎥 Gemini YouTube Q&A Chatbot

An AI-powered Streamlit application that allows users to interact with any YouTube video using natural language. It transcribes the video, performs sentiment analysis, and builds a Q&A chatbot using Google Gemini and FAISS vector store. Perfect for students, educators, and researchers who want to engage with video content more deeply.

---

## 🚀 Features

- 🔗 **YouTube Integration**: Accepts video URLs or IDs to fetch and analyze transcripts.
- 📜 **Transcript Extraction**: Automatically pulls and formats transcripts from YouTube.
- 😃 **Sentiment & Tone Analysis**: Provides polarity and subjectivity metrics using TextBlob.
- 📥 **Download Transcript**: Export transcript as `.txt` or `.pdf`.
- 🧠 **Gemini Embedding + FAISS**: Generates embeddings from the transcript using Gemini and indexes with FAISS.
- 💬 **Interactive Q&A Chatbot**: Ask questions about the video with style customization (e.g., summarization, examples, etc.)
- 🧩 **Style Options for Answers**: Choose from "Teacher Mode", "Bullet Points", "Summary", and more.
- 💻 **Built with Streamlit**: User-friendly and interactive UI.

---

## 📁 Project Structure

```
├── ui.py                # Main Streamlit app UI
├── config.py            # Loads Gemini API key from .env
├── embedding_store.py   # Embedding logic and FAISS integration
├── ingestion.py         # Transcript fetching (assumed module)
├── retrieval_chain.py   # Gemini RAG chain builder (assumed module)
├── .env                 # Environment file containing API key
├── requirements.txt     # All required libraries to install
```

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/zazanali/youtube-gemini-chatbot
cd youtube-gemini-chatbot
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your `.env` file

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Launch the application

```bash
streamlit run ui.py
```

---

## 🛠️ Tech Stack

| Component       | Technology                        |
|----------------|------------------------------------|
| Embeddings      | Google Gemini `text-embedding-004` |
| Vector Store    | FAISS                              |
| UI              | Streamlit                          |
| Transcript Fetch| YouTube Transcript API             |
| LLM Integration | LangChain                          |
| Sentiment Tool  | TextBlob                           |
| PDF Export      | FPDF                               |

---

## 📋 Usage Guide

1. **Paste YouTube URL or ID**.
2. **Click “Load Transcript”** to fetch and analyze.
3. **View & download** the transcript (TXT/PDF).
4. **Check sentiment/tone**.
5. **Chat** with the video using customizable answer styles.
6. **Explore Q&A History** for previous interactions.

---

## 📤 Output Example

**Tone Analysis**
```
- Overall Tone: Neutral
- Polarity Score: 0.05
- Subjectivity Score: 0.47
```

**Q&A Example**
```
Q: What was the main topic of the video?
A: The video mainly discusses how to use Streamlit to build AI tools.
```

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 🙋 Author

**Developed by Zazan Ali**  
📧 Email: alizazan3@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/zazanali)
