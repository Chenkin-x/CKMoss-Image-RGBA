"""Export the merged RGBA VAE and a native ComfyUI DiT LoRA. No optimizer states."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from peft.tuners.tuners_utils import BaseTunerLayer


def export_vae(vae, output):
    # Merge into each original layer, then remove the adapter wrappers entirely.
    for name, layer in list(vae.named_modules()):
        if isinstance(layer, BaseTunerLayer):
            layer.merge(safe_merge=True)
            parent, _, child = name.rpartition('.')
            setattr(vae.get_submodule(parent), child, layer.get_base_layer())
    state = {key: value.detach().cpu().contiguous() for key, value in vae.state_dict().items()}
    config = {key: value for key, value in dict(vae.config).items() if not key.startswith('_')}
    save_file(state, str(output / 'moss_rgba_vae.safetensors'),
              metadata={'format': 'pt', 'image_channels': 'RGBA',
                        'config': json.dumps(config), 'latent_format': 'flux2'})
    (output / 'vae_config.json').write_text(json.dumps(config, indent=2) + '\n')


def export_dit(adapter, output):
    config = json.loads((adapter / 'adapter_config.json').read_text())
    source = load_file(str(adapter / 'adapter_model.safetensors'))
    state = {}
    for name, spec in config['modules'].items():
        prefix = 'diffusion_model.' + name
        state[prefix + '.lora_A.weight'] = source[name + '.lora_A']
        state[prefix + '.lora_B.weight'] = source[name + '.lora_B']
        state[prefix + '.alpha'] = torch.tensor(float(spec['alpha']))
    save_file(state, str(output / 'moss_rgba_dit_lora.safetensors'),
              metadata={'format': 'pt', 'base_model': 'OpenMOSS-Team/MOSS-Image-SFT',
                        'task': 'text-to-rgba', 'vae': 'moss_rgba_vae.safetensors'})
    return len(config['modules'])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vae-config', required=True)
    parser.add_argument('--vae-weights', required=True)
    parser.add_argument('--vae-adapter', required=True)
    parser.add_argument('--dit-adapter', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from moss_rgba.vae import load_adapted_vae, decode_rgba

    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    vae = load_adapted_vae(args.vae_config, args.vae_weights, args.vae_adapter, 'cuda')
    latent = torch.randn(1, 128, 8, 8, device='cuda', generator=torch.Generator(device='cuda').manual_seed(47))
    before = decode_rgba(vae, latent)
    export_vae(vae, out)
    after = decode_rgba(vae, latent)
    error = (before - after).abs()
    assert torch.isfinite(after).all() and error.mean() < 1e-4
    modules = export_dit(Path(args.dit_adapter), out)
    report = {'vae_merge_mean_abs_error': error.mean().item(),
              'vae_merge_max_abs_error': error.max().item(), 'dit_lora_modules': modules,
              'files': {}}
    for path in sorted(out.glob('*.safetensors')):
        with path.open('rb') as f:
            report['files'][path.name] = {'bytes': path.stat().st_size,
                                         'sha256': hashlib.file_digest(f, 'sha256').hexdigest()}
    (out / 'weights.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
