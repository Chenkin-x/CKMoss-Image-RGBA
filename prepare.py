"""Cache captions and fixed RGBA VAE latents in separate GPU passes."""
import argparse
import hashlib
import json
from pathlib import Path

from accelerate import Accelerator
import torch

from moss_rgba.components import load_text_components
from moss_rgba.cache_vae import load_rgba_vae, rgba_pixels, encode_rgba, vae_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dit.json')
    parser.add_argument('--stage', choices=['text', 'latents'], required=True)
    parser.add_argument('--vae-adapter')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    acc = Accelerator()
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in Path(cfg['captions']).read_text().splitlines()]
    root = Path(cfg['cache_dir']); root.mkdir(parents=True, exist_ok=True)
    folder = root / args.stage; folder.mkdir(exist_ok=True)
    if args.stage == 'text':
        provenance = {'snapshot_dir': cfg['snapshot_dir'], 'max_sequence_length': cfg['max_sequence_length']}
        model = load_text_components(cfg['snapshot_dir'], device=acc.device)
        model.text_encoder.to(acc.device)

        @torch.no_grad()
        def encode(batch):
            prompts = [model.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['caption']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True) for row in batch]
            tokens = model.tokenizer(prompts, return_tensors='pt', padding=True,
                                     truncation=True, max_length=cfg['max_sequence_length']).to(acc.device)
            values = model.text_encoder(**tokens, use_cache=False, return_dict=True).last_hidden_state
            return [x[mask.bool()].cpu().contiguous() for x, mask in zip(values, tokens.attention_mask)]

        if acc.is_main_process:
            torch.save(encode([{'caption': ''}])[0], folder / 'null.pt')
    else:
        if args.vae_adapter is None:
            parser.error('--stage latents requires a fixed --vae-adapter checkpoint')
        provenance = {**vae_identity(cfg, args.vae_adapter), 'max_edge': cfg['max_edge']}
        model = load_rgba_vae(cfg, args.vae_adapter, acc.device)

        def encode(batch):
            pixels = rgba_pixels(Path(cfg['image_root']) / batch[0]['path'], cfg['max_edge'])
            z = encode_rgba(model, pixels[None].to(acc.device))[0]
            return [z.to(torch.bfloat16).cpu().contiguous()]

    metadata = folder / 'metadata.json'
    if metadata.exists():
        assert json.loads(metadata.read_text()) == provenance, 'Use a new cache for a different encoder.'
    if acc.is_main_process:
        metadata.write_text(json.dumps(provenance, indent=2))
    acc.wait_for_everyone()
    local = rows[acc.process_index::acc.num_processes]
    batch_size = cfg['text_encode_batch_size'] if args.stage == 'text' else 1
    for begin in range(0, len(local), batch_size):
        pending = []
        for row in local[begin:begin + batch_size]:
            path = folder / f"{row['img_id']:09d}.pt"
            digest = hashlib.sha256(row['caption'].encode()).hexdigest()
            if path.exists():
                saved = torch.load(path, weights_only=True, map_location='cpu')
                assert saved['image_sha256'] == row['sha256']
                if args.stage == 'text':
                    assert saved['caption_sha256'] == digest
            else:
                pending.append(row)
        if pending:
            for row, value in zip(pending, encode(pending)):
                path = folder / f"{row['img_id']:09d}.pt"
                digest = hashlib.sha256(row['caption'].encode()).hexdigest()
                tmp = path.with_suffix('.tmp')
                torch.save({'value': value, 'image_sha256': row['sha256'], 'caption_sha256': digest}, tmp)
                tmp.replace(path)
        end = min(begin + batch_size, len(local))
        if begin // 50 != end // 50 or end == len(local):
            print(f'rank {acc.process_index}: {args.stage} {end}/{len(local)}', flush=True)
    acc.wait_for_everyone()
    if acc.is_main_process:
        (root / 'index.json').write_text(json.dumps([
            {'id': row['img_id'], 'split': row['split'], 'caption': row['caption'],
             'text': f"text/{row['img_id']:09d}.pt", 'latent': f"latents/{row['img_id']:09d}.pt"}
            for row in rows], ensure_ascii=False))
    acc.end_training()


if __name__ == '__main__':
    main()
