import runpod
import torch
import base64
import io
import soundfile as sf
from diffusers import StableAudioPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Audio model...")

pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16
)

pipe = pipe.to(device)

print("Stable Audio model loaded.")

def generate(job):
    prompt = job["input"].get("prompt", "lofi hip hop beat")
    duration = job["input"].get("duration", 10)

    print("Generating audio:", prompt)

    audio = pipe(
        prompt,
        num_inference_steps=200,
        audio_end_in_s=duration
    ).audios[0]

    buffer = io.BytesIO()
    sf.write(buffer, audio, 44100, format="WAV")

    audio_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "audio": audio_base64
    }

runpod.serverless.start(
    {"handler": generate}
)
