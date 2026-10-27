"""Frozen RGBA F2VAE for MOSS text-to-RGBA LoRA adaptation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, AvifImagePlugin
import torch
from torch import nn
from torch.nn import functional as F

AvifImagePlugin.DEFAULT_MAX_THREADS = 1


def vae_identity(config, adapter):
    path = Path(adapter) / 'adapter_model.safetensors'
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'base_config': config['vae_config'], 'base_weights': config['vae_weights'],
            'adapter': str(Path(adapter).resolve()), 'adapter_sha256': digest,
            'packing': 'pixel_unshuffle_2_then_original_bn', 'image_channels': 'RGBA'}


@torch.no_grad()
def load_rgba_vae(config, adapter, device):
    from diffusers import AutoencoderKLFlux2
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    # Use the SAME base VAE as the RGBA LoRA. The MOSS snapshot's slim decoder
    # is a different VAE and cannot load this adapter.
    vae = AutoencoderKLFlux2.from_config(json.loads(Path(config['vae_config']).read_text()))
    vae.load_state_dict(load_file(config['vae_weights']), strict=True)
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


def rgba_pixels(path, max_edge):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGBA')
    scale = min(1, max_edge / max(image.size))
    size = tuple(max(16, round(side * scale / 16) * 16) for side in image.size)
    image = image.resize(size, Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    return pixels.float().div_(127.5).sub_(1)


@torch.no_grad()
def encode_rgba(vae, pixels):
    # Raw F2 latent: C=32, f8. Packed DiT latent: C=128, f16.
    z = F.pixel_unshuffle(vae.encode(pixels).latent_dist.mode(), 2).float()
    mean = vae.bn.running_mean[None, :, None, None].float()
    std = (vae.bn.running_var[None, :, None, None].float() + vae.config.batch_norm_eps).sqrt()
    return (z - mean) / std


@torch.no_grad()
def decode_rgba(vae, latent):
    mean = vae.bn.running_mean[None, :, None, None].float()
    std = (vae.bn.running_var[None, :, None, None].float() + vae.config.batch_norm_eps).sqrt()
    z = F.pixel_shuffle(latent.float() * std + mean, 2)
    return vae.decode(z).sample


def save_rgba(pixels, path):
    pixels = ((pixels.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(path)
