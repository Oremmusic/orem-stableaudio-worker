import runpod
import base64
import torch
import numpy as np
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Audio model...")

model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)

print("Model loaded")

def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt", "melodic trap loop")
    duration = job_input.get("duration", 16)

    conditioning = [{
        "prompt": prompt,
        "seconds_total": duration
    }]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=duration * model.sample_rate
    )

    audio = output.cpu().numpy()[0]

    audio_bytes = audio.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    return {"audio": audio_base64}


runpod.serverless.start({"handler": handler})
