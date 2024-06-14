import math
import torch
from tqdm import tqdm
from einops import repeat
import os
from safetensors.torch import save_file
from safetensors import safe_open

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

def get_timestep_stats(args, accelerator, unet, text_encoder, tokenizer, noise_scheduler, steps=50):
    stats_filename = f"{os.path.basename(args.pretrained_model_name_or_path)}-target_stats.safetensors"
    if os.path.exists(stats_filename):
        with safe_open(stats_filename, framework="pt", device=accelerator.device) as f:
            std_target_by_ts = f.get_tensor("std")
            mean_target_by_ts = f.get_tensor("mean")
    else:
        std_target_by_ts, mean_target_by_ts = generate_timestep_stats(accelerator, unet, text_encoder, tokenizer, noise_scheduler)
        save_file({
            "std": std_target_by_ts.contiguous(),
            "mean": mean_target_by_ts.contiguous()
        }, stats_filename)
    return std_target_by_ts.view(-1, 4, 1, 1), mean_target_by_ts.view(-1, 4, 1, 1)


def generate_timestep_stats(accelerator, unet, text_encoder, tokenizer, scheduler, steps=50, clip_skip=-1):
    logger.info("Generating noise stats for model...this is going to take a moment")
    unet = accelerator.unwrap_model(unet)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    step_stds  = [[] for i in range(0, steps)]
    step_means = [[] for i in range(0, steps)]
    def callback(step, noise_pred):
        step_stds[step].append(noise_pred.std(dim=(0, 2, 3)))
        step_means[step].append(noise_pred.mean(dim=(0, 2, 3)))

    with torch.no_grad():
        unet.set_use_memory_efficient_attention(True, False)
        for prompt in tqdm(DEFAULT_PROMPTS[:10]):
            run_sdxl(prompt, unet, steps,
                        guidance_scale=7.5,
                        scheduler=scheduler,
                        tokenizer1=tokenizer[0],
                        text_model1=text_encoder[0],
                        tokenizer2=tokenizer[1],
                        text_model2=text_encoder[1],
                        height=1024,
                        width=1024,
                        device=accelerator.device,
                        callback=callback)

    std_means  = torch.stack([torch.stack(s).mean(dim=0) for s in step_stds])
    mean_means = torch.stack([torch.stack(s).mean(dim=0) for s in step_means])
    full_stds = torch.nn.functional.interpolate(std_means.unsqueeze(0).permute(0, 2, 1), size=1000).permute(0, 2, 1).squeeze(0).flip(0)
    full_means = torch.nn.functional.interpolate(mean_means.unsqueeze(0).permute(0, 2, 1), size=1000).permute(0, 2, 1).squeeze(0).flip(0)

    return full_stds, full_means

def run_sdxl(prompt, unet, steps, guidance_scale, scheduler, tokenizer1, text_model1, tokenizer2, text_model2, height, width, device, callback, top=0, left=0):
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
    uc_ctx, uc_ctx_pool = call_text_encoder("", "")
    uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

    text_embeddings = torch.cat([uc_ctx, c_ctx])
    vector_embeddings = torch.cat([uc_vector, c_vector])

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
    scheduler.set_timesteps(steps, device)

    # このへんはDiffusersからのコピペ
    # Copy from Diffusers
    timesteps = scheduler.timesteps.to(device)
    num_latent_input = 2

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        callback(i, noise_pred)

        # compute the previous noisy sample x_t -> x_t-1
        # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
