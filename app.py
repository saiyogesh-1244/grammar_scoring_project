
import streamlit as st
import torch
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import time

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from faster_whisper import WhisperModel

# Project paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from grammar_scoring_engine import GrammarScoringModel, AudioDataset

load_dotenv(BASE_DIR / ".env")

# Streamlit configuration
st.set_page_config(
    page_title="Multilingual Grammar Scoring AI",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 AI Multilingual Grammar Scoring System")

st.write(
    "Upload your audio to detect its language, "
    "transcribe speech, identify grammar mistakes "
    "and receive corrections."
)

MODEL_PATH = BASE_DIR / "best_model.pth"


# Load Whisper
@st.cache_resource
def load_whisper():
    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )


# Load existing scoring model
@st.cache_resource
def load_model():
    model = GrammarScoringModel(input_channels=32)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# Gemini multilingual grammar correction
def correct_grammar(text, language_code):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a professional multilingual grammar teacher.

The detected language code is: {language_code}.

Analyze the following speech transcription.

Original transcription:
{text}

Tasks:
1. Identify the important grammar mistakes.
2. Correct the text in the SAME language.
3. Do not translate the corrected text.
4. Explain mistakes briefly in English.
5. Preserve the original meaning.
6. Keep the response short and clear.
7. Do not invent words or sentences.

Use this format:

CORRECTED TEXT:
Write the corrected text in the detected language.

GRAMMAR MISTAKES:
List the mistakes found.

EXPLANATION:
Explain the mistakes in simple English.
"""

    last_error = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config={
                    "http_options": {
                        "timeout": 180000
                    }
                }
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as e:

            last_error = e
            error_message = str(e).upper()

            is_temporary_error = (
    "503" in error_message
    or "UNAVAILABLE" in error_message
    or "429" in error_message
    or "RESOURCE_EXHAUSTED" in error_message
    or "504" in error_message
    or "DEADLINE_EXCEEDED" in error_message
)

            if is_temporary_error and attempt < 2:

                wait_time = 3 * (attempt + 1)

                st.warning(
                    f"Gemini is busy. Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                break

    raise RuntimeError(
        f"Gemini grammar checking failed after retries: {last_error}"
    )


# Audio upload
audio_file = st.file_uploader(
    "Upload your audio recording",
    type=["wav", "mp3", "m4a"]
)

if audio_file is not None:

    st.audio(audio_file)

    if st.button("Analyse Audio", type="primary"):

        temp_path = None

        try:

            # Save uploaded audio temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(audio_file.name).suffix
            ) as temp_file:

                temp_file.write(audio_file.getvalue())
                temp_path = Path(temp_file.name)

            # Speech recognition
            with st.spinner(
                "Detecting language and transcribing audio..."
            ):

                whisper = load_whisper()

                segments, info = whisper.transcribe(
                    str(temp_path),
                    beam_size=5
                )

                transcription = " ".join(
                    segment.text.strip()
                    for segment in segments
                ).strip()

            if not transcription:

                st.error("No speech detected in this audio.")

            else:

                # Automatic language detection
                language_code = info.language

                st.subheader("1. Language Detection")

                st.success(
                    f"Detected language: {language_code}"
                )

                st.write(
                    f"Language confidence: "
                    f"{info.language_probability:.2%}"
                )

                if info.language_probability < 0.60:
                    st.warning(
                        "Language detection confidence is low. "
                        "Please verify the detected language."
                    )

                # Original transcription
                st.subheader("2. Original Transcription")
                st.write(transcription)

                # Grammar correction
                with st.spinner(
                    "Checking grammar using Gemini AI..."
                ):

                    correction = correct_grammar(
                        transcription,
                        language_code
                    )

                st.subheader("3. Grammar Correction")
                st.markdown(correction)

                # Existing grammar score
                st.subheader("4. Grammar Score")

                with st.spinner(
                    "Calculating estimated audio score..."
                ):

                    dataset = AudioDataset(
                        df=pd.DataFrame({
                            "filename": [temp_path.name]
                        }),
                        audio_dir=str(temp_path.parent)
                    )

                    features, _ = dataset[0]

                    model = load_model()

                    with torch.no_grad():

                        prediction = model(
                            features.unsqueeze(0)
                        ).item()

                    score = float(
                        np.clip(prediction, 1, 5)
                    )

                st.metric(
                    "Estimated Audio Score",
                    f"{score:.2f} / 5"
                )

                st.info(
                    "The existing scoring model was trained "
                    "on English data. Its score is not validated "
                    "for other languages. Grammar correction "
                    "is performed separately using Gemini AI."
                )

                st.success(
                    "Audio analysis completed successfully!"
                )

        except Exception as e:

            st.error(f"Analysis failed: {e}")

        finally:

            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
