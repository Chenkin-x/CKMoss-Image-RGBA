"""RGBA conversion and AlphaVAE-style objectives for the FLUX.2 VAE."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from diffusers import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from safetensors.torch import load_file, save_file
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


def load_rgb_vae(config, weights):
    vae = AutoencoderKLFlux2.from_config(json.loads(Path(config).read_text()))
    vae.load_state_dict(load_file(weights), strict=True)
    return vae


@torch.no_grad()
def convert_to_rgba(vae):
    """Keep RGB initialization, latent shape, quant convolutions and BN statistics."""
    conv_in, conv_out = vae.encoder.conv_in, vae.decoder.conv_out
    assert conv_in.in_channels == 3 and conv_out.out_channels == 3
    new_in = nn.Conv2d(4, conv_in.out_channels, conv_in.kernel_size,
                      stride=conv_in.stride, padding=conv_in.padding,
                      device=conv_in.weight.device, dtype=conv_in.weight.dtype)
    new_out = nn.Conv2d(conv_out.in_channels, 4, conv_out.kernel_size,
                       stride=conv_out.stride, padding=conv_out.padding,
                       device=conv_out.weight.device, dtype=conv_out.weight.dtype)
    new_in.weight.zero_()
    new_in.weight[:, :3].copy_(conv_in.weight)
    new_in.bias.copy_(conv_in.bias)
    new_out.weight.zero_()
    new_out.weight[:3].copy_(conv_out.weight)
    new_out.bias[:3].copy_(conv_out.bias)
    new_out.bias[3] = 1.0  # All four image channels use [-1, 1]; alpha=1 is opaque.
    vae.encoder.conv_in, vae.decoder.conv_out = new_in, new_out
    vae.register_to_config(in_channels=4, out_channels=4)
    return vae


def add_rgba_lora(vae, rank=32, alpha=32):
    """Learn the RGBA conversion through Conv2d/Linear LoRA; all base weights stay fixed."""
    targets = [name for name, module in vae.named_modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                        lora_dropout=0.0, bias='none', init_lora_weights=True)
    vae.requires_grad_(False)
    return inject_adapter_in_model(config, vae)


def save_rgba_lora(vae, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vae.peft_config['default'].save_pretrained(directory)
    vae.save_config(directory)
    weights = {key: value.detach().cpu().contiguous()
               for key, value in get_peft_model_state_dict(vae).items()}
    save_file(weights, directory / 'adapter_model.safetensors')


def load_rgba_lora(config, weights, directory):
    vae = convert_to_rgba(load_rgb_vae(config, weights))
    vae = inject_adapter_in_model(LoraConfig.from_pretrained(directory), vae)
    state = load_file(str(Path(directory) / 'adapter_model.safetensors'))
    result = set_peft_model_state_dict(vae, state)
    assert not result.unexpected_keys
    assert not [key for key in result.missing_keys if 'lora_' in key]
    return vae.eval().requires_grad_(False)


def composite(rgba, background):
    """RGBA and background are normalized to [-1, 1]; alpha is converted to [0, 1]."""
    rgb, alpha = rgba[:, :3], (rgba[:, 3:] + 1) * 0.5
    return rgb * alpha + background * (1 - alpha)


def abmse(pred, target, background_mean, background_second_moment):
    pred, target = pred.float(), target.float()
    a_pred, a_target = (pred[:, 3:] + 1) * 0.5, (target[:, 3:] + 1) * 0.5
    p = pred[:, :3] * a_pred - target[:, :3] * a_target
    da = a_pred - a_target
    return (p.square() - 2 * background_mean * p * da
            + background_second_moment * da.square()).mean()


def posterior_kl(posterior, reference=None):
    """KL per latent element. FP32 and expm1 avoid low-precision cancellation."""
    mean, logvar = posterior.mean.float(), posterior.logvar.float()
    if reference is None:
        return 0.5 * (mean.square() + torch.expm1(logvar) - logvar).mean()
    ref_mean, ref_logvar = reference.mean.float(), reference.logvar.float()
    delta = logvar - ref_logvar
    return 0.5 * ((mean - ref_mean).square() * torch.exp(-ref_logvar)
                  + torch.expm1(delta) - delta).mean()


def save_rgba(image, path):
    """Clamp only for image export, never before the training losses."""
    from PIL import Image

    pixels = ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    Image.fromarray(pixels.to(torch.uint8).permute(1, 2, 0).numpy()).save(path)


class RGBATrainingObjective(nn.Module):
    def __init__(self, rgb_vae, loss_cfg, lora_cfg, gradient_checkpointing=False):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.reference = None
        if loss_cfg.ref_kl_weight:
            # Only the old RGB encoder is needed. Do not keep a second decoder.
            self.reference = copy.deepcopy(nn.Sequential(rgb_vae.encoder, rgb_vae.quant_conv))
            self.reference.eval().requires_grad_(False)
        self.vae = add_rgba_lora(convert_to_rgba(rgb_vae), lora_cfg.rank, lora_cfg.alpha)
        if gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
        self.perceptual = None
        if loss_cfg.lpips_weight:
            import lpips

            self.perceptual = lpips.LPIPS(net='vgg', verbose=False).eval().requires_grad_(False)
        self.register_buffer('background_mean',
                             torch.tensor(loss_cfg.background_mean).view(1, 3, 1, 1))
        self.register_buffer('background_second_moment',
                             torch.tensor(loss_cfg.background_second_moment).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.vae.bn.eval()  # Keep the pretrained latent normalization for downstream models.
        if self.reference is not None:
            self.reference.eval()
        if self.perceptual is not None:
            self.perceptual.eval()
        return self

    def forward(self, rgba):
        target = rgba.float()
        dtype = next(self.vae.parameters()).dtype
        inputs = target
        if self.reference is not None:
            # The compatibility branch receives opaque composites, not transparent RGBA.
            opaque_rgb = torch.cat([composite(target, -1), composite(target, 1)])
            opaque = torch.cat([opaque_rgb, torch.ones_like(opaque_rgb[:, :1])], dim=1)
            inputs = torch.cat([target, opaque])

        moments = self.vae.encode(inputs.to(dtype)).latent_dist.parameters.float()
        posterior = DiagonalGaussianDistribution(moments[:len(target)])
        pred = self.vae.decode(posterior.sample().to(dtype)).sample.float()
        reconstruction = abmse(pred, target, self.background_mean.float(),
                               self.background_second_moment.float())
        kl = posterior_kl(posterior)
        perceptual, ref_kl = reconstruction.new_zeros(()), reconstruction.new_zeros(())

        if self.perceptual is not None:
            # In [-1, 1], black is -1, not 0 (which would composite onto gray).
            pred_rgb = torch.cat([composite(pred, -1), composite(pred, 1)])
            target_rgb = torch.cat([composite(target, -1), composite(target, 1)])
            p_dtype = next(self.perceptual.parameters()).dtype
            with torch.autocast(target.device.type, dtype=p_dtype,
                                enabled=p_dtype in (torch.float16, torch.bfloat16)):
                perceptual = self.perceptual(pred_rgb, target_rgb).float().mean()

        if self.reference is not None:
            with torch.no_grad():
                ref_dtype = next(self.reference.parameters()).dtype
                teacher = DiagonalGaussianDistribution(self.reference(opaque_rgb.to(ref_dtype)).float())
            student = DiagonalGaussianDistribution(moments[len(target):])
            ref_kl = posterior_kl(student, teacher)

        # Each term is averaged over its own elements, rather than scaled by image CHW.
        # These coefficients are configurable starting points, not a claim of paper-equivalent scaling.
        terms = dict(abmse=reconstruction, lpips=perceptual * self.loss_cfg.lpips_weight,
                     kl=kl * self.loss_cfg.kl_weight, ref_kl=ref_kl * self.loss_cfg.ref_kl_weight)
        loss = sum(terms.values())
        logs = {f'loss/{key}': value.detach() for key, value in terms.items()}
        logs.update({'loss/total': loss.detach(), 'raw/lpips': perceptual.detach(),
                     'raw/kl': kl.detach(), 'raw/ref_kl': ref_kl.detach(),
                     'alpha/mae': (pred[:, 3:].detach() - target[:, 3:]).abs().mean() * 0.5,
                     'latent/rms': posterior.mean.detach().square().mean().sqrt()})
        return loss, logs
