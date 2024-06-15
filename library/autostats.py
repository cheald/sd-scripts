import ast
import math
import torch
from tqdm import tqdm
from einops import repeat
import os
from safetensors.torch import save_file
from safetensors import safe_open
import numpy as np

from library.utils import setup_logging, add_logging_arguments

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

def get_timestep_stats(args, accelerator, unet, text_encoder, tokenizer, noise_scheduler):
    stats_filename = args.autostats
    timesteps = TIMESTEPS
    if os.path.exists(stats_filename):
        with safe_open(stats_filename, framework="pt") as f:
            std_means = f.get_tensor("std")
            mean_means = f.get_tensor("mean")
            timesteps = f.get_tensor("timesteps").tolist()
    else:
        autostats_kwargs = {}
        if args.autostats_args is not None and len(args.autostats_args) > 0:
            for arg in args.autostats_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                autostats_kwargs[key] = value

        std_means, mean_means = generate_timestep_stats(accelerator, unet, text_encoder, tokenizer, noise_scheduler, height=args.resolution[1], width=args.resolution[0], **autostats_kwargs)
        save_file({
            "std": std_means.contiguous(),
            "mean": mean_means.contiguous(),
            "timesteps": torch.tensor(TIMESTEPS),
        }, stats_filename)

    std_target_by_ts = interp(std_means, timesteps).to(device=accelerator.device, dtype=torch.float32).view(-1, 4, 1, 1)
    mean_target_by_ts = interp(mean_means, timesteps).to(device=accelerator.device, dtype=torch.float32).view(-1, 4, 1, 1)

    return std_target_by_ts, mean_target_by_ts


def generate_timestep_stats(accelerator, unet, text_encoder, tokenizer, scheduler, steps=50, clip_skip=-1, width=1024, height=1024, prompts=10):
    logger.info("Generating noise stats for model...this is going to take a moment")
    unet = accelerator.unwrap_model(unet)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    steps = len(TIMESTEPS)

    step_stds  = [[] for i in range(0, steps)]
    step_means = [[] for i in range(0, steps)]
    def callback(step, noise_pred):
        step_stds[step].append(noise_pred.std(dim=(0, 2, 3)))
        step_means[step].append(noise_pred.mean(dim=(0, 2, 3)))

    with torch.no_grad():
        with tqdm(total=prompts*steps) as pbar:
            for prompt in DEFAULT_PROMPTS[:prompts]:
                run_sdxl(prompt, unet, steps,
                            guidance_scale=7.5,
                            scheduler=scheduler,
                            tokenizer1=tokenizer[0],
                            text_model1=text_encoder[0],
                            tokenizer2=tokenizer[1],
                            text_model2=text_encoder[1],
                            height=height,
                            width=width,
                            device=accelerator.device,
                            callback=callback,
                            pbar=pbar)

    std_means  = torch.stack([torch.stack(s).mean(dim=0) for s in step_stds])
    mean_means = torch.stack([torch.stack(s).mean(dim=0) for s in step_means])

    return std_means, mean_means

def run_sd15():
    pass


def run_sdxl(prompt, unet, steps, guidance_scale, scheduler, tokenizer1, text_model1, tokenizer2, text_model2, height, width, device, callback, pbar, top=0, left=0):
    def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
                device=timesteps.device
            )
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, "b -> b d", d=dim)
        return embedding


    def get_timestep_embedding(x, outdim):
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        # x = rearrange(x, "b d -> (b d)")
        x = torch.flatten(x)
        emb = timestep_embedding(x, outdim)
        # emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=outdim)
        emb = torch.reshape(emb, (b, dims * outdim))
        return emb

    emb1 = get_timestep_embedding(torch.FloatTensor([height, width]).unsqueeze(0), 256)
    emb2 = get_timestep_embedding(torch.FloatTensor([top, left]).unsqueeze(0), 256)
    emb3 = get_timestep_embedding(torch.FloatTensor([height, width]).unsqueeze(0), 256)

    c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    uc_vector = c_vector.clone().to(device)

    def call_text_encoder(text, text2):
        # text encoder 1
        batch_encoding = tokenizer1(
            text,
            truncation=True,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(device)

        with torch.no_grad():
            enc_out = text_model1(tokens, output_hidden_states=True, return_dict=True)
            text_embedding1 = enc_out["hidden_states"][11]
            # text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)    # layer normは通さないらしい

        # text encoder 2
        # tokens = tokenizer2(text2).to(DEVICE)
        tokens = tokenizer2(
            text,
            truncation=True,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(device)

        with torch.no_grad():
            enc_out = text_model2(tokens, output_hidden_states=True, return_dict=True)
            text_embedding2_penu = enc_out["hidden_states"][-2]
            # logger.info("hidden_states2", text_embedding2_penu.shape)
            text_embedding2_pool = enc_out["text_embeds"]  # do not support Textual Inversion

        # 連結して終了 concat and finish
        text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)
        return text_embedding, text_embedding2_pool

    # cond
    c_ctx, c_ctx_pool = call_text_encoder(prompt, prompt)
    # logger.info(c_ctx.shape, c_ctx_p.shape, c_vector.shape)
    c_vector = torch.cat([c_ctx_pool, c_vector], dim=1)

    # uncond
    # uc_ctx, uc_ctx_pool = call_text_encoder("", "")
    # uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

    text_embeddings = torch.cat([c_ctx])
    vector_embeddings = torch.cat([c_vector])

    latents_shape = (1, 4, height // 8, width // 8)
    latents = torch.randn(
        latents_shape,
        generator=None,
        device="cpu",
        dtype=torch.float32,
    ).to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma

    # set timesteps
    # scheduler.set_timesteps(steps, device)
    scheduler.timesteps = torch.tensor(TIMESTEPS, device=device)

    # このへんはDiffusersからのコピペ
    # Copy from Diffusers
    timesteps = scheduler.timesteps.to(device)
    num_latent_input = 1

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)
        pbar.update(1)

        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        callback(i, noise_pred)

        # compute the previous noisy sample x_t -> x_t-1
        # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
