"""Extract required objects from pinned official ZIP using HTTP byte ranges."""
import io
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/objects.zip"

class RemoteFile(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        response = self.session.get(URL, headers={"Range": "bytes=-65536"}, timeout=60)
        response.raise_for_status()
        assert response.status_code == 206
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.url = response.url
        self.position = 0
        self.cache = [(self.size - len(response.content), response.content)]
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position
    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size,self.size-self.position)
        if size <= 0: return b""
        for start, data in self.cache:
            if start <= self.position and self.position+size <= start+len(data):
                result = data[self.position-start:self.position-start+size]
                self.position += size
                return result
        response = self.session.get(self.url, headers={"Range":f"bytes={self.position}-{self.position+size-1}"}, timeout=120)
        response.raise_for_status()
        assert response.status_code == 206 and len(response.content) == size, response.headers
        self.cache.append((self.position,response.content))
        self.position += size
        return response.content

remote = RemoteFile()
with zipfile.ZipFile(remote) as archive:
    files = [i for i in archive.infolist() if "/060_kitchenpot/" in i.filename]
    assert files
    print("Required assets",len(files),"compressed bytes",sum(i.compress_size for i in files),flush=True)
    # Prefetch contiguous ZIP entries together instead of thousands of requests.
    start = min(i.header_offset for i in files)
    end = max(i.header_offset + i.compress_size + len(i.filename.encode()) + 4096 for i in files)
    remote.seek(start)
    remote.read(end - start)
    files += [archive.getinfo(name) for name in ["objects/objaverse/list.json", "objects/same.json"]]
    for info in files:
        target = ROOT / "simulation/RoboTwin/assets" / info.filename
        assert target.resolve().is_relative_to((ROOT / "simulation/RoboTwin/assets").resolve())
        if info.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(info))
        print(info.filename,flush=True)
