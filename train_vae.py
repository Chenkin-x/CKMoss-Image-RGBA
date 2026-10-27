"""RGBA LoRA fine-tuning of F2VAE with Accelerate / ZeRO-2.

accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 4 \
    train_vae.py --config configs/vae.yaml --output outputs/flux2-rgba \
    data.root=/path/to/rgba
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from moss_rgba.data import RGBADataset, init_worker
from moss_rgba.vae_training import RGBATrainingObjective, load_rgb_vae, save_rgba, save_rgba_lora, composite


@torch.no_grad()
def evaluate(model, loader, accelerator, output, step):
    vae = accelerator.unwrap_model(model).vae
    vae.eval()
    totals = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    preview_paths = []
    preview = output / 'previews' / f'{step:08d}'
    for batch in loader:
        with accelerator.autocast():
            pred = vae(batch.to(next(vae.parameters()).dtype), sample_posterior=False).sample.float().clamp(-1, 1)
        target = batch.float()
        psnr = []
        for background in (-1, 1):
            mse = ((composite(pred, background) - composite(target, background)) * 0.5).square().flatten(1).mean(1)
            psnr.append(-10 * mse.clamp_min(1e-12).log10())
        alpha_mae = ((pred[:, 3:] - target[:, 3:]) * 0.5).abs().flatten(1).mean(1)
        totals += torch.stack([psnr[0].sum(), psnr[1].sum(), alpha_mae.sum(), target.new_tensor(len(target))])
        if accelerator.is_main_process and len(preview_paths) < 8:
            preview.mkdir(parents=True, exist_ok=True)
            for original, restored in zip(target, pred):
                index = len(preview_paths) // 2
                for suffix, image in [('input', original), ('reconstruction', restored)]:
                    path = preview / f'{index}-{suffix}.png'
                    save_rgba(image, path)
                    preview_paths.append(path)
                if len(preview_paths) >= 8:
                    break
    totals = accelerator.reduce(totals, reduction='sum').cpu().tolist()
    metrics = {'eval/psnr_black': totals[0]/totals[3], 'eval/psnr_white': totals[1]/totals[3],
               'eval/alpha_mae': totals[2]/totals[3], 'eval/images': totals[3]}
    metrics['eval/psnr'] = (metrics['eval/psnr_black'] + metrics['eval/psnr_white']) * 0.5
    if accelerator.is_main_process:
        with (output / 'validation.jsonl').open('a') as file:
            file.write(json.dumps(dict(step=step, **metrics)) + '\n')
        if accelerator.log_with:
            import wandb
            metrics['eval/reconstructions'] = [wandb.Image(str(path), caption=path.stem) for path in preview_paths]
    accelerator.log(metrics, step=step)
    model.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vae.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', help='checkpoint directory, including its state/ subdirectory')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=cfg.training.gradient_accumulation_steps, sync_with_dataloader=False),
        mixed_precision=cfg.training.mixed_precision,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.training.seed),
        log_with='wandb' if cfg.logging.wandb else None,
    )
    if accelerator.state.deepspeed_plugin is not None:
        ds = accelerator.state.deepspeed_plugin.deepspeed_config
        ds['gradient_accumulation_steps'] = cfg.training.gradient_accumulation_steps
        ds['train_micro_batch_size_per_gpu'] = cfg.data.batch_size
        ds['gradient_clipping'] = 0.0
    set_seed(cfg.training.seed)
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / 'config.yaml')
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        rgb_vae = load_rgb_vae(cfg.model.config, cfg.model.weights)
        objective = RGBATrainingObjective(rgb_vae, cfg.loss, cfg.model.lora,
                                         cfg.model.gradient_checkpointing)
    dataset = RGBADataset(cfg.data.root, cfg.data.resolution, cfg.data.blend_probability)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers, pin_memory=True,
                        persistent_workers=cfg.data.num_workers > 0, drop_last=True,
                        worker_init_fn=init_worker)
    val_data = RGBADataset(cfg.data.validation_root, cfg.data.resolution, 0, training=False)
    val_loader = DataLoader(val_data, batch_size=cfg.data.validation_batch_size,
                           num_workers=2, pin_memory=True, worker_init_fn=init_worker)
    trainable = [p for p in objective.vae.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.optimizer.lr,
                                 betas=tuple(cfg.optimizer.betas), weight_decay=cfg.optimizer.weight_decay)
    model, optimizer, loader = accelerator.prepare(objective, optimizer, loader)
    val_loader = accelerator.prepare_data_loader(val_loader)

    step, epoch, next_batch = 0, 0, 0
    if args.resume:
        checkpoint = Path(args.resume)
        accelerator.load_state(checkpoint / 'state')
        progress = json.loads((checkpoint / 'progress.json').read_text())
        step, epoch, next_batch = progress['step'], progress['epoch'], progress['next_batch']

    wandb_state = None
    if cfg.logging.wandb:
        options = dict(name=cfg.logging.name, entity=cfg.logging.entity, dir=str(output))
        if args.resume:
            previous = json.loads((Path(args.resume) / 'wandb-run.json').read_text())
            options.update(id=previous['id'], resume='must')
        accelerator.init_trackers(cfg.logging.project, config=OmegaConf.to_container(cfg, resolve=True),
                                  init_kwargs={'wandb': options})
        if accelerator.is_main_process:
            run = accelerator.get_tracker('wandb', unwrap=True)
            wandb_state = dict(id=run.id, url=run.url, project=run.project, entity=run.entity)
            (output / 'wandb-run.json').write_text(json.dumps(wandb_state, indent=2))

    def save():
        checkpoint = output / f'checkpoint-{step:08d}'
        checkpoint.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(checkpoint / 'state')
        if accelerator.is_main_process:
            # Keep inference adapters separate from optimizer/resume state.
            save_rgba_lora(accelerator.unwrap_model(model).vae, checkpoint / 'lora')
            (checkpoint / 'progress.json').write_text(json.dumps(
                dict(step=step, epoch=epoch, next_batch=next_batch), indent=2))
            if wandb_state:
                (checkpoint / 'wandb-run.json').write_text(json.dumps(wandb_state, indent=2))
        accelerator.wait_for_everyone()

    global_batch = cfg.data.batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    accelerator.print(f'RGBA images={len(dataset)}, global batch={global_batch}, start step={step}, '
                      f'trainable LoRA parameters={sum(p.numel() for p in trainable):,}')
    if accelerator.is_main_process:
        runtime = dict(world_size=accelerator.num_processes, global_batch=global_batch,
                       micro_batch=cfg.data.batch_size, accumulation=cfg.training.gradient_accumulation_steps,
                       trainable_parameters=sum(p.numel() for p in trainable),
                       trainable_only_lora=all('lora_' in name for name, p in objective.vae.named_parameters() if p.requires_grad),
                       zero_stage=model.zero_optimization_stage() if accelerator.state.deepspeed_plugin else None)
        (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    if step == 0:
        evaluate(model, val_loader, accelerator, output, step)
    model.train()
    sums, count, logged_step = {}, 0, step
    started = time.perf_counter()
    while step < cfg.training.max_steps:
        loader.set_epoch(epoch)
        batches = accelerator.skip_first_batches(loader, next_batch) if next_batch else loader
        for batch_index, rgba in enumerate(batches, start=next_batch):
            with accelerator.accumulate(model):
                warmup = cfg.training.warmup_steps
                if step < warmup:
                    lr_scale = (step + 1) / warmup
                else:
                    progress = (step - warmup) / max(1, cfg.training.max_steps - warmup)
                    lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group['lr'] = cfg.optimizer.lr * lr_scale
                loss, logs = model(rgba)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in logs.items():
                sums[key] = sums.get(key, 0) + value.float()
            count += 1
            next_batch = batch_index + 1
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % cfg.training.log_every == 0:
                keys = sorted(sums)
                values = accelerator.reduce(torch.stack([sums[k] / count for k in keys]), 'mean')
                metrics = dict(zip(keys, values.cpu().tolist()))
                metrics.update(step=step, lr=optimizer.param_groups[0]['lr'],
                               seconds_per_microbatch=(time.perf_counter() - started) / count,
                               seconds_per_step=(time.perf_counter() - started) / (step-logged_step),
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                accelerator.log(metrics, step=step)
                if accelerator.is_main_process:
                    with (output / 'metrics.jsonl').open('a') as file:
                        file.write(json.dumps(metrics) + '\n')
                    print(f"step={step} loss={metrics['loss/total']:.6f} "
                          f"abmse={metrics['loss/abmse']:.6f} alpha_mae={metrics['alpha/mae']:.5f} "
                          f"s/step={metrics['seconds_per_step']:.3f}", flush=True)
                sums, count = {}, 0
                logged_step = step
                started = time.perf_counter()
            if step % cfg.training.validate_every == 0:
                evaluate(model, val_loader, accelerator, output, step)
                started = time.perf_counter()
            if cfg.training.save_every and (step % cfg.training.save_every == 0 or step == cfg.training.max_steps):
                save()
                started = time.perf_counter()
            if step == cfg.training.max_steps:
                break
        epoch += 1
        next_batch = 0
    accelerator.end_training()


if __name__ == '__main__':
    main()
