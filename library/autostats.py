import torch
from tqdm import tqdm
import os
from safetensors.torch import save_file
from safetensors import safe_open
import numpy as np
from library.utils import setup_logging
from gen_img import PipelineLike
import random

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "A photo of a cat",
    "An artistic drawing of a woman",
    "A fantasy landscape at night",
    "A bustling cyberpunk city at high noon",
    "A high resolution stock photo",
    "A picture of a serene mountain landscape",
    "A street scene from a small medieval village",
    "A close up of a sunset over the ocean",
    "A portrait of a person from the 1920s",
    "A busy cityscape at rush hour",
    "An abstract design with vibrant colors",
    "A black and white photo of a horse in a field",
    "A comic book style illustration of a superhero",
    "A still life of fruit in a bowl",
    "A photo of a dense jungle",
    "A photo of a snowy winter scene",
    "A detailed illustration of a mechanical device",
    "A watercolor painting of a field of flowers",
    "A minimalist black and white cityscape",
    "A surrealist painting with distorted perspectives",
    "A photo of a crowded market in a foreign city",
    "A photo of a single tree in an open field",
    "A photo of a old, dusty library",
    "A photorealistic illustration of an animal",
    "A detailed pencil sketch of a person",
    "A pop art style illustration of an iconic celebrity",
    "A photo of a construction site",
    "A photo of a group of musicians performing",
    "A photo of a person working in a laboratory",
    "A photo of a person gardening",
    "A photo of a person in a business meeting",
    "A photo of a factory from the industrial revolution",
    "A photo of a modern, eco-friendly building",
    "A photo of a person hiking in the mountains",
    "A photo of a person practicing yoga.",
]

# The curve is much sharper at lower timesteps, so we sample them at a higher resolution.
TIMESTEPS = [
        999,979,958,938,918,897,877,856,836,816,795,775,755,734,714,693,673,653,
        632,612,592,571,551,531,510,490,469,449,429,408,388,368,347,327,307,286,
        266,245,225,205,184,164,144,123,103,82,62,42,31,21,19,17,15,13,11,9,8,7,
        6,5,4,3,2,1
]

def interp(t, timesteps):
    p = t.permute(1, 0)
    ch = []
    for i in range(0, 4):
        x = list(np.interp(np.arange(0, 1000), list(reversed(timesteps)), p[i].flip(0).cpu().numpy()))
        ch.append( torch.tensor(x) )
    return torch.stack(ch).permute(1, 0).to(t.device)

def get_timestep_stats(args, accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler, is_sdxl):
    stats_filename = args.autostats
    timesteps = TIMESTEPS
    if os.path.exists(stats_filename):
        with safe_open(stats_filename, framework="pt") as f:
            std_means = f.get_tensor("std")
            mean_means = f.get_tensor("mean")
            timesteps = f.get_tensor("timesteps").tolist()
    else:
        prompts = random.sample(DEFAULT_PROMPTS, 16)
        if args.autostats_prompts:
            prompts = args.autostats_prompts
        std_means, mean_means = generate_timestep_stats(
            accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler, is_sdxl,
            height=args.resolution[1],
            width=args.resolution[0],
            batch_size=args.batch_size,
            prompts=prompts
        )

        save_file({
            "std": std_means.contiguous(),
            "mean": mean_means.contiguous(),
            "timesteps": torch.tensor(TIMESTEPS),
        }, stats_filename)

    std_target_by_ts = interp(std_means.to(dtype=torch.float32), timesteps).to(device=accelerator.device, dtype=torch.float32).view(-1, 4, 1, 1)
    mean_target_by_ts = interp(mean_means.to(dtype=torch.float32), timesteps).to(device=accelerator.device, dtype=torch.float32).view(-1, 4, 1, 1)

    return std_target_by_ts, mean_target_by_ts


def generate_timestep_stats(accelerator, unet, vae, text_encoder, tokenizer, scheduler, is_sdxl, steps=50, clip_skip=1, width=1024, height=1024, batch_size=1, prompts=[]):
    logger.info("Collecting noise stats for model. This may take a while...")
    unet = accelerator.unwrap_model(unet)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = [accelerator.unwrap_model(text_encoder)]

    if isinstance(tokenizer, (list, tuple)):
        tokenizer = [accelerator.unwrap_model(t) for t in tokenizer]
    else:
        tokenizer = [accelerator.unwrap_model(tokenizer)]

    steps = len(TIMESTEPS)
    step_stds  = [None for i in range(0, steps)]
    step_means = [None for i in range(0, steps)]

    pipeline = PipelineLike(is_sdxl, accelerator.device, vae, text_encoder, tokenizer, unet, scheduler, clip_skip=clip_skip)

    with torch.no_grad(), accelerator.autocast(), tqdm(total=len(prompts)*steps) as pbar:
        def callback(step, timestep, noise_pred):
            pbar.update(noise_pred.shape[0])
            stds = noise_pred.std(dim=(2, 3)).to(dtype=torch.float32)
            means = noise_pred.mean(dim=(2, 3)).to(dtype=torch.float32)
            if step_stds[step] is None:
                step_stds[step] = stds
            else:
                step_stds[step] = torch.cat([step_stds[step], stds], dim=0)
            if step_means[step] is None:
                step_means[step] = means
            else:
                step_means[step] = torch.cat([step_means[step], means], dim=0)

        chunked_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        for prompt in chunked_prompts:
            pipeline(
                prompt=prompt,
                height=height,
                width=width,
                manual_timesteps=TIMESTEPS,
                guidance_scale=7.5,
                noise_callback=callback,
                return_latents=True
            )

    std_means  = torch.stack([s.mean(dim=0) for s in step_stds])
    mean_means = torch.stack([s.mean(dim=0) for s in step_means])

    del pipeline

    return std_means, mean_means
