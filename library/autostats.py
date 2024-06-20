import torch
import torch.nn.functional as F
import os
from safetensors import safe_open
import numpy as np

from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

standard_normal_distribution = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
def smooth(probs, step_size=0.5):
    kernel = standard_normal_distribution.log_prob(torch.arange(-torch.pi, torch.pi, step_size) ).exp().to(probs.device)
    smoothed = F.conv1d(probs[None, None, :].float(), kernel[None, None, :].float(), padding="same").reshape(-1)
    return smoothed / smoothed.sum()

def kerras_timesteps(n, sigma_min=0.001, sigma_max=10.0):
    alpha_min = torch.arctan(torch.tensor(sigma_min))
    alpha_max = torch.arctan(torch.tensor(sigma_max))
    step_indices = torch.arange(n)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas

# cribbed from A111
def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}
        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
             logger.error(f"Error reading metadata from file: {filename}", exc_info=True)

        return res

def interp_forward(t, timesteps):
    p = t.permute(1, 0).float().cpu().numpy() # Switch to channel-first and flip the order from first-denoised to first-noised
    rev_ts = torch.tensor(timesteps).tolist() # Reverse the timesteps from denoising order to noising order
    xs = np.arange(0, 1000)
    t = torch.stack([torch.tensor(list(np.interp(xs, rev_ts, p[i]))) for i in range(0, 4)])
    return t.permute(1, 0).to(t.device)

def load_model_noise_stats(args):
    if args.autostats is None or not os.path.exists(args.autostats):
        return None, None
    with safe_open(args.autostats, framework="pt") as f:
        observations = f.get_tensor("observations")
        timesteps = f.get_tensor("timesteps")
    return transform_observations(observations, timesteps)

def transform_observations(observations, timesteps):
    # shape is [timestep, sample, channels, h, w]
    # we average on sample, h, w so that we get stats for [timestep, channel]

    means = observations.mean(dim=(1, 3, 4))
    stds = observations.std(dim=(1, 3, 4))
    return interp_forward(means, timesteps), interp_forward(stds, timesteps)

def autostats(args, generator):
    timestep_probs = torch.ones(1000)
    std_target_by_ts = mean_target_by_ts = scaled_std_target_by_ts = scaled_mean_target_by_ts = None

    mean_target_by_ts, std_target_by_ts = load_model_noise_stats(args)
    if mean_target_by_ts is None:
        generator()
        mean_target_by_ts, std_target_by_ts = load_model_noise_stats(args)

    if mean_target_by_ts is None:
        raise ValueError("Could not load noise stats from model")

    std_target_by_ts = std_target_by_ts.view(-1, 4, 1, 1)
    mean_target_by_ts = mean_target_by_ts.view(-1, 4, 1, 1)

    std_weighting = (std_target_by_ts - 1).abs()
    std_weighting = std_weighting / std_weighting.max(dim=0).values

    mean_weighting = mean_target_by_ts.abs()
    mean_weighting = mean_weighting / mean_weighting.max(dim=0).values

    effect_scale = args.autostats_true_noise_weight
    scaled_std_target_by_ts = (std_target_by_ts - 1.0) * effect_scale[0] + 1.0
    scaled_mean_target_by_ts = (mean_target_by_ts * effect_scale[1])

    if args.autostats_timestep_weighting:
        timestep_probs = (std_target_by_ts - 1).abs().mean(dim=1).reshape(-1)
        timestep_probs[:15] = timestep_probs[15]
        timestep_probs = smooth(timestep_probs)

    timestep_probs = timestep_probs / timestep_probs.sum()

    print("std", scaled_std_target_by_ts.view(-1, 4))
    print("mean", scaled_mean_target_by_ts.view(-1, 4))

    return std_target_by_ts, mean_target_by_ts, scaled_std_target_by_ts, scaled_mean_target_by_ts, timestep_probs
