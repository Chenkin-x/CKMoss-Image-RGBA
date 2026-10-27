"""MOSS text-to-RGBA LoRA, fixed VAE caches, Accelerate and ZeRO-2."""
import argparse
import json
from pathlib import Path
import time
import gc

from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, set_seed
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
import torch
from torch.utils.data import DataLoader, Dataset

from moss_rgba.model import MossPackedModel
from moss_rgba.components import load_transformer, load_moss_module
from moss_rgba.lora import inject_lora, load_adapter, save_adapter
from moss_rgba.cache_vae import load_rgba_vae, decode_rgba, save_rgba


@torch.no_grad()
def sample_previews(model, acc, cfg, output, step):
    from PIL import Image, ImageDraw

    model = acc.unwrap_model(model)
    model.eval()
    root = Path(cfg['cache_dir'])
    index = {r['id']: r for r in json.loads((root / 'index.json').read_text())}
    null = torch.load(root / 'text/null.pt', weights_only=True).to(acc.device)
    vae = load_rgba_vae(cfg, cfg['vae_adapter'], acc.device)
    scheduler = load_moss_module(cfg['snapshot_dir']).MOSSImageScheduler.from_pretrained(
        Path(cfg['snapshot_dir']) / 'scheduler', local_files_only=True)
    folder = output / 'samples' / f'{step:06d}'; folder.mkdir(parents=True, exist_ok=True)
    for i, image_id in enumerate(cfg['sample_ids']):
        if i % acc.num_processes != acc.process_index:
            continue
        row = index[image_id]
        target = torch.load(root / row['latent'], weights_only=True)['value']
        text = torch.load(root / row['text'], weights_only=True)['value'].to(acc.device)
        generator = torch.Generator(device=acc.device).manual_seed(cfg['seed'] + i)
        z = torch.randn(target.shape, device=acc.device, generator=generator)
        scheduler.set_timesteps(cfg['sample_steps'], device=acc.device)
        for t in scheduler.timesteps:
            positive, negative = model([z.to(torch.bfloat16)] * 2, (t.float() / 1000).expand(2),
                                       [text, null])
            v = negative.float() + cfg['sample_cfg'] * (positive.float() - negative.float())
            z = scheduler.step(v[None], t, z[None], return_dict=False)[0][0].float()
        image = decode_rgba(vae, z[None])[0]
        path = folder / f'{image_id}.png'; save_rgba(image, path)
        # The saved model output remains RGBA; checkerboards are only log previews.
        rgba = Image.open(path).convert('RGBA')
        checker = Image.new('RGB', rgba.size, (192, 192, 192)); draw = ImageDraw.Draw(checker)
        for y in range(0, rgba.height, 32):
            for x in range(0, rgba.width, 32):
                if (x // 32 + y // 32) % 2:
                    draw.rectangle((x, y, x + 31, y + 31), fill=(224, 224, 224))
        checker.paste(rgba, mask=rgba.getchannel('A'))
        checker.save(folder / f'{image_id}-preview.jpg')
    del vae
    gc.collect(); torch.cuda.empty_cache()
    acc.wait_for_everyone()
    if acc.is_main_process and cfg['wandb']['enabled']:
        import wandb
        acc.log({'sample/rgba': [wandb.Image(str(folder / f'{i}-preview.jpg'),
                  caption=index[i]['caption']) for i in cfg['sample_ids']]}, step=step)
    model.train()


class CachedRGBA(Dataset):
    def __init__(self, directory):
        self.root = Path(directory)
        self.rows = [row for row in json.loads((self.root / 'index.json').read_text())
                     if row['split'] == 'train']
        self.null = torch.load(self.root / 'text/null.pt', weights_only=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        z = torch.load(self.root / row['latent'], weights_only=True)['value']
        text = torch.load(self.root / row['text'], weights_only=True)['value']
        return z, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dit.json')
    parser.add_argument('--resume')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    accum = cfg['gradient_accumulation_steps']
    ds = {
        'train_micro_batch_size_per_gpu': cfg['micro_batch_size'],
        'train_batch_size': cfg['global_batch_size'],
        'gradient_accumulation_steps': accum,
        'bf16': {'enabled': True}, 'gradient_clipping': 0,
        'zero_optimization': {'stage': 2, 'overlap_comm': True, 'contiguous_gradients': True,
                              'reduce_bucket_size': 25000000, 'allgather_bucket_size': 25000000},
        'zero_allow_untested_optimizer': True,
        'steps_per_print': 1000000000,
    }
    acc = Accelerator(mixed_precision='bf16', deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds),
                      gradient_accumulation_plugin=GradientAccumulationPlugin(
                          num_steps=accum, sync_with_dataloader=False),
                      dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg['seed']),
                      log_with='wandb' if cfg['wandb']['enabled'] else None)
    assert cfg['global_batch_size'] == acc.num_processes * cfg['micro_batch_size'] * accum
    set_seed(cfg['seed']); torch.set_num_threads(4)
    base = load_transformer(cfg['snapshot_dir'], device='cpu', dtype=torch.bfloat16)
    inject_lora(base, rank=cfg['lora_rank'], alpha=cfg['lora_alpha'],
                targets=cfg['lora_targets'], param_dtype=torch.bfloat16)
    model = MossPackedModel(base, gradient_checkpointing=cfg['gradient_checkpointing'])
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.use_muon = True
    optimizer = MuonWithAuxAdam([dict(params=params, use_muon=True, lr=cfg['learning_rate'],
                                     momentum=.95, weight_decay=cfg['weight_decay'],
                                     ns_method='gram', adjust_lr_fn='match_rms_adamw')])
    dataset = CachedRGBA(cfg['cache_dir'])
    loader = DataLoader(dataset, batch_size=cfg['micro_batch_size'], shuffle=True, drop_last=True,
                        num_workers=cfg['num_workers'], pin_memory=True, collate_fn=list,
                        persistent_workers=cfg['num_workers'] > 0)
    model, optimizer, loader = acc.prepare(model, optimizer, loader)
    null = dataset.null.to(acc.device)
    output = Path(cfg['output_dir']); output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'task': 'text_to_rgba', 'snapshot_dir': cfg['snapshot_dir'],
        'vae': json.loads((Path(cfg['cache_dir']) / 'latents/metadata.json').read_text()),
        'lora_targets': cfg['lora_targets'], 'config': cfg,
    }
    if acc.is_main_process:
        (output / 'config.json').write_text(json.dumps(cfg, indent=2))
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        runtime = {'world_size': acc.num_processes, 'zero_stage': model.zero_optimization_stage(),
                   'global_batch_size': cfg['global_batch_size'], 'micro_batch_size': cfg['micro_batch_size'],
                   'gradient_accumulation_steps': accum, 'lora_rank': cfg['lora_rank'],
                   'trainable_parameters': sum(p.numel() for p in params),
                   'total_parameters': sum(p.numel() for p in base.parameters()),
                   'trainable_only_lora': all('lora_' in n for n, p in base.named_parameters() if p.requires_grad),
                   'optimizer': 'DeepSpeed MuonWithAuxAdam', 'adjust_lr_fn': 'match_rms_adamw',
                   'training_images': len(dataset), 'vae_adapter': cfg['vae_adapter']}
        (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        print(json.dumps(runtime), flush=True)
    step, epoch, consumed_batches = 0, 0, 0
    if args.resume:
        # Strictly check the saved LoRA. Frozen base weights are deliberately
        # absent from the ZeRO checkpoint and have already been reloaded above.
        load_adapter(base, Path(args.resume) / 'lora')
        acc.load_state(Path(args.resume) / 'state', load_module_strict=False)
        state = json.loads((Path(args.resume) / 'progress.json').read_text())
        step, consumed_batches = state['step'], state['consumed_batches']
        epoch, skip = divmod(consumed_batches, len(loader))
        if acc.is_main_process:
            (output / 'SAVE_AND_PAUSE').unlink(missing_ok=True)
    else:
        skip = 0
    acc.wait_for_everyone()
    if acc.is_main_process:
        (output / 'status.json').write_text(json.dumps({'state': 'running', 'started': time.time(), 'start_step': step}))
    if cfg['wandb']['enabled']:
        options = {k: cfg['wandb'][k] for k in ['entity', 'name']}
        options['dir'] = str(output)
        if args.resume:
            run = json.loads((output / 'wandb-run.json').read_text())
            options.update(id=run['id'], resume='must')
        acc.init_trackers(cfg['wandb']['project'], config=cfg, init_kwargs={'wandb': options})
        if acc.is_main_process:
            run = acc.get_tracker('wandb', unwrap=True)
            (output / 'wandb-run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
    if cfg['sample_at_start'] and step == 0:
        acc.print('Generating fixed validation previews at step 0.', flush=True)
        sample_previews(model, acc, cfg, output, step)
        acc.print('Step 0 previews saved; starting DiT LoRA updates.', flush=True)
    start = time.monotonic(); initial_step = step
    total_loss = torch.zeros((), device=acc.device)
    stop = False
    model.train()
    while step < cfg['max_steps'] and not stop:
        loader.set_epoch(epoch)
        batches = acc.skip_first_batches(loader, skip) if skip else loader
        skip = 0
        for batch in batches:
            lr = cfg['learning_rate'] * min(1., (step + 1) / cfg['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            with acc.accumulate(model):
                # t=0 is clean, t=1 is noise. Latent prediction uses v = noise - clean.
                t = (torch.randn(len(batch), device=acc.device) * cfg['time_std']
                     + cfg['time_mean']).sigmoid().clamp(1e-5, 1 - 1e-5)
                images, targets, contexts = [], [], []
                for i, (z, context) in enumerate(batch):
                    z = z.to(acc.device, dtype=torch.float32)
                    noise = torch.randn_like(z)
                    images.append(((1 - t[i]) * z + t[i] * noise).to(torch.bfloat16))
                    targets.append(noise - z)
                    contexts.append(null if torch.rand((), device=acc.device) < cfg['cond_dropout']
                                    else context.to(acc.device))
                pred = model(images, t, contexts)
                # Per-image mean: large images do not receive a larger loss weight.
                loss = torch.stack([(p.float() - v).square().mean() for p, v in zip(pred, targets)]).mean()
                acc.backward(loss)
                optimizer.step(); optimizer.zero_grad()
                total_loss += loss.detach() / accum
            consumed_batches += 1
            if acc.sync_gradients:
                step += 1
                status = torch.stack([total_loss, total_loss.new_tensor((output / 'SAVE_AND_PAUSE').exists())])
                logged, stop = acc.reduce(status, reduction='mean').tolist()
                stop = bool(stop); total_loss.zero_()
                acc.log({'train/loss': logged, 'train/lr': lr}, step=step)
                if step % 10 == 0:
                    seconds = (time.monotonic() - start) / (step - initial_step)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    acc.log({'train/seconds_per_step': seconds, 'train/peak_allocated_gib': peak}, step=step)
                    acc.print(f'step={step} loss={logged:.6f} lr={lr:.3g} s/step={seconds:.3f} peak_gib={peak:.2f}')
                    if acc.is_main_process:
                        with (output / 'metrics.jsonl').open('a') as f:
                            f.write(json.dumps({'step': step, 'loss': logged, 'lr': lr,
                                               'seconds_per_step': seconds, 'peak_allocated_gib': peak}) + '\n')
                if step % cfg['save_every'] == 0 or step == cfg['max_steps'] or stop:
                    folder = output / f'checkpoint-{step:06d}'
                    acc.save_state(folder / 'state', exclude_frozen_parameters=True)
                    if acc.is_main_process:
                        save_adapter(acc.unwrap_model(model).base_model, folder / 'lora', metadata)
                        (folder / 'progress.json').write_text(json.dumps(
                            {'step': step, 'consumed_batches': consumed_batches}))
                    acc.wait_for_everyone()
                if step % cfg['sample_every'] == 0 and not stop:
                    sample_previews(model, acc, cfg, output, step)
                if step == cfg['max_steps'] or stop:
                    break
        epoch += 1
    if acc.is_main_process:
        (output / 'status.json').write_text(json.dumps({'state': 'paused' if stop else 'complete', 'step': step}))
    acc.end_training()


if __name__ == '__main__':
    main()
