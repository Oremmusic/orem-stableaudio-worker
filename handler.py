import runpod
import torch
import base64
import io
import soundfile as sf

from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Audio model...")

# Load Stable Audio model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

print("Stable Audio model loaded successfully")


def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt", "melodic trap drum loop")
    bpm = job_input.get("bpm", 120)
    bars = job_input.get("bars", 4)

    # Calculate accurate loop duration
    seconds_per_beat = 60 / bpm
    beats_per_bar = 4
    duration = bars * beats_per_bar * seconds_per_beat

    print(f"Generating {bars} bar loop at {bpm} BPM")
    print(f"Duration: {duration} seconds")
    print(f"Prompt: {prompt}")

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    # Generate audio
    output = generate_diffusion_cond(
        model,
        steps=40,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Rearrange audio batch
    output = rearrange(output, "b d n -> d (b n)")
    audio = output.cpu().numpy()

    # Encode WAV to memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio.T, sample_rate, format="WAV")

    audio_base64 = base64.b64encode(buffer.getvalue()).decode()

    print("Generation complete")

    return {
        "audio": audio_base64,
        "sample_rate": sample_rate,
        "bpm": bpm,
        "bars": bars,
        "duration": duration
    }


runpod.serverless.start({"handler": handler})
