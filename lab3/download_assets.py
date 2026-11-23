"""实验准备辅助：官方服务器的分段下载，校验后才生成完整文件。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import time
from urllib.parse import unquote, urlparse
from urllib.request import Request, ProxyHandler, build_opener

urlopen = build_opener(ProxyHandler({})).open

ROOT = Path(__file__).resolve().parent
WHEEL = "torch-2.7.1+cu126-cp311-cp311-win_amd64.whl"


class WheelLinks(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if unquote(urlparse(href).path).endswith("/" + WHEEL):
                self.link = href


def file_hash(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, target, algorithm, expected, workers):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target, algorithm) != expected:
            raise RuntimeError(f"Existing target checksum mismatch: {target}")
        print(f"Already verified: {target.name}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".part")
    prefix = partial.stat().st_size if partial.exists() else 0
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
        header = r.headers.get("Content-Range", "")
        if r.status != 206 or not re.fullmatch(r"bytes 0-0/\d+", header):
            raise RuntimeError("Server does not provide verifiable byte ranges")
        total = int(header.split("/")[-1])
    if prefix > total:
        raise RuntimeError("Partial file larger than expected")
    if prefix == total:
        if file_hash(partial, algorithm) != expected:
            raise RuntimeError("Complete partial file failed checksum")
        partial.replace(target)
        return
    size = (total - prefix + workers - 1) // workers
    ranges = [(i, start, min(start + size - 1, total - 1))
              for i, start in enumerate(range(prefix, total, size))]
    started = time.perf_counter()

    def part(item):
        i, start, end, path = item
        if path.exists() and path.stat().st_size == end - start + 1:
            return i, path
        for attempt in range(3):
            try:
                have = 0
                if path.exists():
                    with path.open("rb") as previous:
                        have = previous.seek(0, 2)
                if have == end - start + 1:
                    return i, path
                if have > end - start + 1:
                    raise RuntimeError("Segment prefix exceeds expected size")
                request_start = start + have
                with urlopen(Request(url, headers={"Range": f"bytes={request_start}-{end}"}), timeout=120) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {request_start}-{end}/{total}":
                        raise RuntimeError("Unexpected Content-Range")
                    received = 0
                    with path.open("ab") as f:
                        for block in iter(lambda: r.read(1024 * 1024), b""):
                            f.write(block)
                            received += len(block)
                    if received + have != end - start + 1:
                        raise RuntimeError("Incomplete range")
                return i, path
            except Exception:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    # 保留早先的大分段前缀，剩余内容拆成 8 MiB 小块。
    pieces, jobs = [], []
    for i, start, end in ranges:
        existing = target.with_suffix(target.suffix + f".range-{i:02d}.part")
        downloaded = existing.stat().st_size if existing.exists() else 0
        if downloaded > end - start + 1:
            raise RuntimeError("Range prefix exceeds expected size")
        if downloaded:
            pieces.append(existing)
        for offset in range(start + downloaded, end + 1, 8 * 1024 * 1024):
            last = min(offset + 8 * 1024 * 1024 - 1, end)
            path = target.with_suffix(target.suffix + f".segment-{offset}.part")
            pieces.append(path)
            jobs.append((len(jobs), offset, last, path))
    connections = min(48, workers * 4)
    print(f"{target.name}: {len(jobs)} remaining blocks; {connections} connections", flush=True)
    paths = {}
    with ThreadPoolExecutor(connections) as pool:
        for future in as_completed([pool.submit(part, item) for item in jobs]):
            i, path = future.result()
            paths[i] = path
            if len(paths) % 10 == 0 or len(paths) == len(jobs):
                print(f"{target.name}: {len(paths)}/{len(jobs)} blocks; {time.perf_counter()-started:.1f} s", flush=True)
    with partial.open("ab") as f:
        for path in pieces:
            with path.open("rb") as source:
                shutil.copyfileobj(source, f, 4 * 1024 * 1024)
    if partial.stat().st_size != total or file_hash(partial, algorithm) != expected:
        raise RuntimeError("Combined download checksum failed; no final target created")
    partial.replace(target)
    for path in set(pieces):
        path.unlink()
    print(f"Verified {target.name}: {algorithm}={expected}; {time.perf_counter()-started:.1f} s", flush=True)


def wheel():
    with urlopen("https://download.pytorch.org/whl/cu126/torch/", timeout=60) as r:
        parser = WheelLinks()
        parser.feed(r.read().decode())
    link = parser.link
    parsed = urlparse(link)
    if parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
        raise RuntimeError("Unexpected wheel host")
    expected = parsed.fragment.removeprefix("sha256=")
    if not re.fullmatch("[0-9a-f]{64}", expected):
        raise RuntimeError("Official wheel SHA-256 missing")
    fetch(link.split("#")[0], ROOT / ".downloads" / WHEEL, "sha256", expected, 12)


def data():
    fetch("https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
          ROOT / "data" / "cifar-10-python.tar.gz", "md5", "c58f30108f718f92721af3b95e74349a", 6)


if __name__ == "__main__":
    import sys
    {"wheel": wheel, "data": data}[sys.argv[1]]()
