"""RGBA F2VAE loading and the original packed latent normalization."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file


@torch.no_grad()
def load_adapted_vae(config_path, weights, adapter, device='cpu'):
    from diffusers import AutoencoderKLFlux2
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils import set_peft_model_state_dict

    vae = AutoencoderKLFlux2.from_config(json.loads(Path(config_path).read_text()))
    vae.load_state_dict(load_file(str(weights)), strict=True)
    source_in, source_out = vae.encoder.conv_in, vae.decoder.conv_out
    conv_in = nn.Conv2d(4, source_in.out_channels, 3, padding=1)
    conv_out = nn.Conv2d(source_out.in_channels, 4, 3, padding=1)
    conv_in.weight.zero_(); conv_in.weight[:, :3].copy_(source_in.weight)
    conv_in.bias.copy_(source_in.bias)
    conv_out.weight.zero_(); conv_out.weight[:3].copy_(source_out.weight)
    conv_out.bias[:3].copy_(source_out.bias); conv_out.bias[3] = 1
    vae.encoder.conv_in, vae.decoder.conv_out = conv_in, conv_out
    vae.register_to_config(in_channels=4, out_channels=4)
    vae = inject_adapter_in_model(LoraConfig.from_pretrained(adapter), vae)
    result = set_peft_model_state_dict(vae, load_file(str(Path(adapter) / 'adapter_model.safetensors')))
    assert not result.unexpected_keys
    assert not [key for key in result.missing_keys if 'lora_' in key]
    return vae.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def load_vae(weights, device='cpu'):
    from diffusers import AutoencoderKLFlux2

    with safe_open(str(weights), framework='pt') as f:
        config = json.loads(f.metadata()['config'])
    vae = AutoencoderKLFlux2.from_config(config)
    vae.load_state_dict(load_file(str(weights)), strict=True)
    return vae.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def encode_rgba(vae, image):
    latent = F.pixel_unshuffle(vae.encode(image).latent_dist.mode(), 2).float()
    mean = vae.bn.running_mean[None, :, None, None].float()
    std = (vae.bn.running_var[None, :, None, None].float() + vae.config.batch_norm_eps).sqrt()
    return (latent - mean) / std


def decode_rgba(vae, latent):
    mean = vae.bn.running_mean[None, :, None, None].float()
    std = (vae.bn.running_var[None, :, None, None].float() + vae.config.batch_norm_eps).sqrt()
    return vae.decode(F.pixel_shuffle(latent.float() * std + mean, 2)).sample


def save_rgba(image, path):
    from PIL import Image

    pixels = ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(path)
