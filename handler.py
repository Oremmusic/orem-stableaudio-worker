import runpod
import torch
import base64
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Audio model...")

model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

print("Model loaded successfully")


def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt", "melodic trap loop")
    duration = job_input.get("duration", 16)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")

    audio = output.cpu().numpy()

    audio_bytes = audio.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    return {"audio": audio_base64}


runpod.serverless.start({"handler": handler})
