# Code entry points

Run commands from the repository root. The project is imported directly; it does not need `pip install -e .`. Dependency versions are in `requirements.txt`. ComfyUI users do not need this environment; use the separate node package.

## VAE

Set original F2VAE weights, data, and output paths in `configs/vae.yaml`. The input files must contain a real alpha channel. `configs/flux2-vae.json` describes the original full F2VAE; the slimmer decoder bundled with MOSS is a different model.

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 4 \
  train_vae.py --config configs/vae.yaml --output outputs/vae
```

The VAE adapter adds an alpha input/output while retaining the original latent layout. The objectives are in `moss_rgba/vae_training.py`. Each saved checkpoint separates the adapter in `lora/` from resume state.

## Captions and frozen caches

`tools/caption.py` uses an OpenAI-compatible vision endpoint. Set `--model` to its served model name. Its manifest is JSONL, with `img_id`, `path`, `split`, `sha256`, and `alpha_min` fields. `path` is relative to `--image-root`; `split` is `train` or `validation`.

```bash
python tools/caption.py --manifest data/rgba/manifest.jsonl \
  --image-root data/rgba --prompt configs/caption.txt \
  --model minicpm-v47 --output data/captions.jsonl
```

The checkerboard is a captioning preview only. Original RGBA pixels remain unchanged. Captions retain manifest fields and add `caption`. Set paths in `configs/dit.json`, including the exact VAE checkpoint selected for adaptation.

```bash
accelerate launch --num_processes 4 --multi_gpu prepare.py --stage text
accelerate launch --num_processes 4 --multi_gpu prepare.py --stage latents \
  --vae-adapter outputs/vae/checkpoint-00001500/lora
```

Use a new latent cache when changing the VAE. Cache metadata pins the adapter digest; it must match `vae_adapter` in the DiT config.

## DiT

The packed model handles variable image/text lengths without buckets. It keeps only the text-to-RGBA path; editing reference masks and reference caching are not included. Its attention dependency is listed explicitly in `requirements.txt`.

The optional Muon source patch is specific to DeepSpeed 0.19.6 and is an explicit setup step, not an import-time modification:

```bash
python tools/patch_deepspeed_muon.py
accelerate launch --multi_gpu --num_processes 4 train_dit.py --config configs/dit.json
```

`train_dit.py` creates the Accelerate / ZeRO-2 configuration. Set per-device batch and accumulation so their product with the GPU count equals `global_batch_size`. VAE and text features remain frozen. The loss is averaged per image. W&B is disabled in the example configuration.

Resume using `--resume outputs/dit/checkpoint-001000`. Create `outputs/dit/SAVE_AND_PAUSE` to save at a complete optimizer update and stop all ranks. This file is cleared only when explicitly resuming.

## Export

```bash
python -m tools.export_weights \
  --vae-config configs/flux2-vae.json \
  --vae-weights models/flux2-vae.safetensors \
  --vae-adapter outputs/vae/checkpoint-00001500/lora \
  --dit-adapter outputs/dit/checkpoint-005602/lora \
  --output release
```

Exports a merged RGBA VAE and native ComfyUI DiT LoRA; optimizer state and base DiT weights are excluded. The exporter compares decoded pixels before/after merging and records file checksums.
