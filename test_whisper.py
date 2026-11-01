
from faster_whisper import WhisperModel

print("Loading Whisper...")
model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

audio_path = input("Enter your audio file path: ")

segments, info = model.transcribe(
    audio_path,
    beam_size=5
)

print("\nDetected language:", info.language)
print("Language probability:", round(info.language_probability, 2))

print("\nTranscribed text:")

for segment in segments:
    print(segment.text)
