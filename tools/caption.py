"""Caption RGBA assets with MiniCPM; append transparency from known alpha.

Run a small manifest with each candidate prompt before captioning the full set.
Existing successful rows are resumed by image ID and prompt hash.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import io
import json
from pathlib import Path
import signal
import threading
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, ImageDraw, AvifImagePlugin

AvifImagePlugin.DEFAULT_MAX_THREADS = 1


def image_views(path, max_edge, view):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
    # Resize RGBA together; never discard alpha before compositing.
    rgba.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    if view == 'checker':
        canvas = Image.new('RGBA', rgba.size, (192, 192, 192, 255))
        draw = ImageDraw.Draw(canvas)
        for y in range(0, rgba.height, 32):
            for x in range(0, rgba.width, 32):
                if (x // 32 + y // 32) % 2:
                    draw.rectangle((x, y, x + 31, y + 31), fill=(224, 224, 224, 255))
        canvases = [canvas]
    else:
        canvases = [Image.new('RGBA', rgba.size, (v, v, v, 255)) for v in (0, 255)]
    views = []
    for canvas in canvases:
        rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        views.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return views


def caption(row, args, prompt, prompt_hash):
    start = time.monotonic()
    views = image_views(Path(args.image_root) / row["path"], args.max_edge, args.view)
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": view}} for view in views]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(args.endpoint.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    choice = result["choices"][0]
    text = choice["message"]["content"].strip()
    # Truncated output must not silently become a training caption.
    if choice["finish_reason"] != "stop" or not text:
        raise RuntimeError(f"Incomplete caption: {choice['finish_reason']}")
    full_caption = text
    if args.transparency_tag and row['alpha_min'] < 255:
        full_caption = text.rstrip('. ') + '. ' + args.transparency_tag + '.'
    return {**row, "caption": full_caption, "caption_body": text, "caption_model": args.model,
            "prompt_sha256": prompt_hash, "caption_seconds": time.monotonic() - start,
            "usage": result.get("usage"), "caption_views": args.view,
            "transparency_tag": args.transparency_tag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30146/v1")
    parser.add_argument("--model", default="minicpm-v47")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--view", choices=['checker', 'black_white'], default='checker')
    parser.add_argument("--transparency-tag", default='transparent background')
    parser.add_argument("--limit", type=int, default=0, help='Limit newly captioned images for throughput measurements.')
    args = parser.parse_args()

    prompt = Path(args.prompt).read_text().strip()
    settings = {'prompt': prompt, 'model': args.model, 'view': args.view,
                'transparency_tag': args.transparency_tag, 'max_edge': args.max_edge,
                'max_tokens': args.max_tokens, 'temperature': 0, 'enable_thinking': False}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            assert row["prompt_sha256"] == digest, "Use a separate output for each prompt."
            completed.add(row["img_id"])
    output.with_suffix('.settings.json').write_text(json.dumps(settings, indent=2))
    pending = [row for row in rows if row["img_id"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    start, errors = time.monotonic(), 0
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    print(json.dumps({"total": len(rows), "complete": len(completed), "pending": len(pending),
                      "prompt_sha256": digest}), flush=True)
    with output.open("a") as handle, output.with_suffix(".errors.jsonl").open("a") as error_file:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            remaining = iter(pending)
            futures = {}
            for _ in range(min(args.workers, len(pending))):
                row = next(remaining)
                futures[pool.submit(caption, row, args, prompt, digest)] = row
            i = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row = futures.pop(future)
                    i += 1
                    try:
                        result = future.result()
                    except Exception as error:
                        errors += 1
                        error_file.write(json.dumps({"img_id": row["img_id"], "error": str(error)}) + "\n")
                        error_file.flush()
                    else:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                    elapsed = time.monotonic() - start
                    print(f"{len(completed) + i - errors}/{len(rows)} captions; "
                          f"errors={errors}; {i / max(elapsed, 1):.3f} images/s", flush=True)
                    row = None if stopping.is_set() else next(remaining, None)
                    if row is not None:
                        futures[pool.submit(caption, row, args, prompt, digest)] = row
    elapsed = time.monotonic() - start
    output.with_suffix('.run.json').write_text(json.dumps({
        'workers': args.workers, 'new_captions': i - errors, 'errors': errors,
        'seconds': elapsed, 'images_per_second': (i - errors) / max(elapsed, 1),
        'stopped': stopping.is_set()}, indent=2))
    if errors:
        raise SystemExit(f"{errors} images failed; see {output.with_suffix('.errors.jsonl')}")


if __name__ == "__main__":
    main()
