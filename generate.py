"""Generate a transparent PNG, with an optional DiT LoRA."""
import argparse
import gc
from pathlib import Path

import torch
from safetensors.torch import load_file

from moss_rgba.components import load_text_components, encode_prompts, load_transformer, load_moss_module
from moss_rgba.vae import load_vae, decode_rgba, save_rgba


@torch.no_grad()
def apply_lora(model, path, strength=1.0):
    state = load_file(path)
    for key, a in state.items():
        if not key.endswith('.lora_A.weight'):
            continue
        prefix = key.removesuffix('.lora_A.weight')
        layer = model.get_submodule(prefix.removeprefix('diffusion_model.'))
        b = state[prefix + '.lora_B.weight']
        scale = strength * float(state[prefix + '.alpha']) / a.shape[0]
        update = b.float() @ a.float()
        layer.weight.add_(update.to(layer.weight.device, layer.weight.dtype), alpha=scale)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Local official MOSS-Image-SFT snapshot')
    parser.add_argument('--vae', required=True, help='moss_rgba_vae.safetensors')
    parser.add_argument('--lora', help='Optional moss_rgba_dit_lora.safetensors')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args()
    assert args.height % 16 == args.width % 16 == 0
    torch.set_num_threads(4)
    device = torch.device('cuda')
    text = load_text_components(args.base, device=device)
    conditions = encode_prompts(text, args.prompt, max_sequence_length=512)
    contexts = [x.embeds[0, x.attention_mask[0]] for x in (conditions.positive, conditions.negative)]
    del text, conditions
    gc.collect(); torch.cuda.empty_cache()

    from moss_rgba.model import MossPackedModel
    base = load_transformer(args.base, device='cpu')
    if args.lora:
        apply_lora(base, args.lora, args.strength)
    model = MossPackedModel(base.to(device)).eval()
    scheduler = load_moss_module(args.base).MOSSImageScheduler.from_pretrained(
        Path(args.base) / 'scheduler', local_files_only=True)
    scheduler.set_timesteps(args.steps, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(128, args.height // 16, args.width // 16, device=device, generator=rng)
    for i, time in enumerate(scheduler.timesteps):
        pred, uncond = model([z.to(torch.bfloat16)] * 2, (time.float() / 1000).expand(2), contexts)
        velocity = uncond.float() + args.cfg * (pred.float() - uncond.float())
        z = scheduler.step(velocity[None], time, z[None], return_dict=False)[0][0].float()
        print(f'{i + 1}/{args.steps}', flush=True)
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    image = decode_rgba(load_vae(args.vae, device), z[None])[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_rgba(image, args.output)


if __name__ == '__main__':
    main()
