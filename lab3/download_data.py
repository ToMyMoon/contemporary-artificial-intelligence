"""从 CIFAR 官方地址下载原始压缩包；断点文件不会当作完整数据使用。"""
import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    directory = ROOT / "data"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "cifar-10-python.tar.gz"
    if target.exists():
        if digest(target) != MD5:
            raise RuntimeError("Existing archive has incorrect MD5; original file left untouched")
        print("Archive already verified", flush=True)
        return
    partial = target.with_suffix(".gz.part")
    start, received, next_report = time.perf_counter(), 0, 16 * 1024 * 1024
    with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as f:
        print(f"Downloading from {response.url}", flush=True)
        for block in iter(lambda: response.read(1024 * 1024), b""):
            f.write(block)
            received += len(block)
            if received >= next_report:
                print(f"{received / 1024 ** 2:.1f} MiB; {time.perf_counter() - start:.1f} s", flush=True)
                next_report += 16 * 1024 * 1024
    if digest(partial) != MD5:
        raise RuntimeError("Downloaded archive failed MD5 validation")
    partial.replace(target)
    print(f"Verified official archive: {received} bytes; MD5 {MD5}", flush=True)


if __name__ == "__main__":
    main()
