<p align="center"><img src="assets/banner.png" alt="CKMoss Image RGBA" width="100%"></p>

<p align="center"><b>Characters, illustrations, and little objects — ready for your composition.</b></p>
<p align="center"><a href="README.md">中文</a> · English · <a href="https://huggingface.co/ChenkinLab/CKMoss-Image-RGBA">Model weights</a> · <a href="https://github.com/Chenkin-x/ComfyUI-CKMoss-Image-RGBA">ComfyUI nodes</a></p>
<p align="center">Author: <a href="https://github.com/chinoll">chinoll</a></p>

CKMoss Image RGBA provides an **RGBA VAE** and an **optional DiT LoRA** for generating images with an alpha channel using MOSS-Image. Create character assets, illustrations, and layers for your compositions.

**You can generate images without the DiT LoRA.** Keep the original MOSS-Image model and use the RGBA VAE. The optional LoRA adapts MOSS-Image-SFT for transparent-image generation; transparency quality can differ between the two combinations.

This repository contains the model code. For image generation in ComfyUI, use the separate [ComfyUI repository](https://github.com/Chenkin-x/ComfyUI-CKMoss-Image-RGBA), including Chinese and English guides and workflows for both combinations.

## Choose a setup

| Setup | Files |
| --- | --- |
| Original MOSS-Image + RGBA VAE | `vae/moss_rgba_vae.safetensors` |
| MOSS-Image-SFT + RGBA VAE + DiT LoRA | The VAE above and `loras/moss_rgba_dit_lora.safetensors` |

Both weights are in [ChenkinLab/CKMoss-Image-RGBA](https://huggingface.co/ChenkinLab/CKMoss-Image-RGBA/tree/main). The VAE is already merged. The LoRA requires a base model. Obtain the base model and its text encoder separately.

## Showcase

Fresh outputs from the RGBA VAE and DiT LoRA, with the original alpha preserved. Background colors and layout are for display only.

<img src="assets/showcase.png" alt="Actual CKMoss Image RGBA outputs" width="100%">

Original PNG files, prompts, and seeds: [assets/samples](assets/samples).

## Generate with Python

Run from the repository root in an environment with the dependencies available; no project installation is needed. `--base` points to a complete local copy of the official MOSS-Image-SFT repository.

```bash
python generate.py \
  --base models/MOSS-Image-SFT \
  --vae models/moss_rgba_vae.safetensors \
  --prompt "A small fox holding a white flower, watercolor illustration, transparent background." \
  --output fox.png
```

This command uses **no DiT LoRA**. To enable it, add:

```bash
  --lora models/moss_rgba_dit_lora.safetensors
```

## Code map

| Entry | Purpose |
| --- | --- |
| `generate.py` | RGBA PNG generation, with optional LoRA |
| `moss_rgba/` | Model, VAE, LoRA, datasets, and objectives |
| `train_vae.py` / `train_dit.py` | Separate VAE and DiT adaptation |
| `prepare.py` | Cache text embeddings and latents |
| `tools/export_weights.py` | Export standalone inference weights |
| `tools/render_showcase.py` | Rebuild the banner and poster from actual PNG outputs |

Developer entry points and configuration: [code guide](docs/code.md).

## Credits and license

Built on [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) and the [FLUX.2 VAE](https://github.com/black-forest-labs/flux2#flux2-autoencoder). Thanks to [niangao2331](https://github.com/niangao2331) for data support.

This repository's code is licensed under [GPLv3](LICENSE). The upstream [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE) and [FLUX.2 VAE](https://github.com/black-forest-labs/flux2#flux2-autoencoder) models are released under Apache 2.0; see the linked upstream license information.
