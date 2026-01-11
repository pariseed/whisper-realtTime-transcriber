import subprocess
import numpy as np
import whisper


SOURCE_NAME = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1

model = whisper.load_model("small.en")
#WHISPER_MODEL = "medium.en"


cmd = [
    "parec",
    "--device", SOURCE_NAME,
    "--format", "float32le",
    "--rate", str(SAMPLE_RATE),
    "--channels", "1"
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

print("Listening to system audio... Ctrl+C to stop\n")

try:
    while True:
        bytes_needed = SAMPLE_RATE * CHUNK_SECONDS * 4
        raw = process.stdout.read(bytes_needed)
        if not raw:
            break

        audio = np.frombuffer(raw, dtype=np.float32)
        if np.sqrt(np.mean(audio**2)) < 0.003:
            continue

        result = model.transcribe(audio, fp16=False)

        text = result["text"].strip()
        if text:
            print(text)

except KeyboardInterrupt:
    print("\nStopping...")
    process.terminate()

