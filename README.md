# 🎙️ Multilingual Grammar Scoring System

An AI-powered application that analyzes spoken audio, converts speech into text, detects language, corrects grammatical mistakes, and predicts a grammar score.

## 🚀 Live Demo

**Try the application here:**

[Grammar Scoring System – Live Application](https://grammarscoringproject-lrxngzxx7g7xfrjpfmtcqp.streamlit.app/)

## 📌 Project Overview

The Multilingual Grammar Scoring System is designed to help users improve their spoken language and grammar through AI-powered speech recognition and grammar correction.

Users can upload an audio recording, receive its transcription, view grammar corrections, and get a predicted grammar score.

## ✨ Features

- 🎤 Audio file upload (WAV, MP3, M4A)
- 🌍 Automatic language detection
- 📝 Speech-to-text transcription
- 🤖 AI-powered grammar correction
- 📊 Automatic grammar score prediction
- 💬 Grammar explanations
- 🌐 Multilingual speech processing

## 🛠️ Technologies Used

- Python
- Streamlit
- PyTorch
- Faster-Whisper
- Google Gemini API
- Librosa
- NumPy
- Pandas
- Scikit-learn

## ⚙️ How It Works

1. Upload an audio recording.
2. The Whisper model converts speech into text.
3. The application detects the spoken language.
4. Google Gemini analyzes and corrects grammar.
5. The trained deep learning model predicts a grammar score.
6. View the transcription, corrections, and score.

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/saiyogesh-1244/grammar_scoring_project.git
```

Navigate to the project folder:

```bash
cd grammar_scoring_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key using `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Run the application:

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
grammar_scoring_project/
│
├── app.py
├── best_model.pth
├── requirements.txt
├── src/
├── .streamlit/
├── index.html
├── script.js
└── style.css
```

## ⚠️ Note

The grammar scoring model was trained on English-language data. Its scores have not been validated for other languages. Multilingual transcription and grammar correction are separate features.

## 👨‍💻 Developer

**Sai Yogesh**

GitHub: [@saiyogesh-1244](https://github.com/saiyogesh-1244)

Project Repository: [Grammar Scoring System](https://github.com/saiyogesh-1244/grammar_scoring_project)

---

⭐ If you find this project useful, consider giving the repository a star!
