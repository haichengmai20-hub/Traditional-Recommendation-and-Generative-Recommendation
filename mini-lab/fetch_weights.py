"""Fetch and verify MiniOneRec's public Industrial checkpoint. Python 3.11+, curl."""
import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "Industrial_ckpt"
TARGET = ROOT / "model.safetensors"
SIZE = 3_555_990_368
SHA = "57531f3d80db89634368829980fbf077938e55a2c5c3f028d943e31e0bf86b55"
REVISION = "365a03fc6601ed36abd69cc9a0a59025a3d31cdc"
URL = f"https://huggingface.co/kkknight/MiniOneRec/resolve/{REVISION}/Industrial_ckpt/model.safetensors"
BLOCK = 32 * 1024 * 1024


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == SIZE and checksum(TARGET) == SHA:
        print("Verified", SHA)
        return
    scratch = ROOT / ".weight-parts"
    scratch.mkdir(exist_ok=True)
    # A stopped ordinary curl transfer can supply the completed initial blocks.
    prefix = TARGET.stat().st_size if TARGET.exists() and TARGET.stat().st_size < SIZE else 0

    def fetch(start):
        end = min(start + BLOCK, SIZE)
        part = scratch / str(start)
        if part.exists() and part.stat().st_size == end - start:
            return part
        temp = part.with_suffix(".new")
        if end <= prefix:
            with TARGET.open("rb") as src, temp.open("wb") as dst:
                src.seek(start)
                dst.write(src.read(end - start))
        else:
            for attempt in range(8):
                saved = temp.stat().st_size if temp.exists() else 0
                if saved == end - start:
                    break
                tail = part.with_suffix(".tail")
                command = ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
                           "--range", f"{start + saved}-{end - 1}", "--write-out", "%{http_code}",
                           "--output", str(tail), URL]
                result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
                if result.stdout.strip() == "206" and tail.exists():
                    if tail.stat().st_size > end - start - saved:
                        raise RuntimeError(f"Unexpected byte range at {start + saved}")
                    with temp.open("ab") as dst, tail.open("rb") as src:
                        shutil.copyfileobj(src, dst)
                tail.unlink(missing_ok=True)
        if temp.stat().st_size != end - start:
            raise RuntimeError(f"Truncated weight range at {start}")
        os.replace(temp, part)
        return part

    starts = list(range(0, SIZE, BLOCK))
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(fetch, start): start for start in starts}
        for completed in as_completed(jobs):
            try:
                completed.result()
            except (OSError, RuntimeError) as error:
                failures.append((jobs[completed], error))
    if failures:
        raise RuntimeError(f"Failed ranges; rerun to resume: {failures}")
    parts = [scratch / str(start) for start in starts]
    staged = scratch / "assembled"
    with staged.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst)
    if checksum(staged) != SHA:
        raise RuntimeError("SHA-256 mismatch; checkpoint was not installed")
    os.replace(staged, TARGET)
    for part in parts:
        part.unlink()
    scratch.rmdir()
    print("Verified", SHA)


if __name__ == "__main__":
    main()
