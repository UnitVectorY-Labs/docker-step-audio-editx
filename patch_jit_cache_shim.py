"""Keep upstream shim code intact, selecting just this image's cache dependency.

FlashInfer's full shim depends on every GPU architecture. Its documented minimal
installation uses only selected providers. Repack that dependency selection so
the image remains small AND requirement validation remains truthful.
"""
import base64
import csv
import hashlib
import io
from pathlib import Path
import platform
import sys
from zipfile import ZipFile

source, destination = map(Path, sys.argv[1:])
architecture = {"aarch64": "sm121a", "x86_64": "sm120f"}[platform.machine()]
selected = "flashinfer-jit-cache-" + architecture
with ZipFile(source) as wheel:
    files = {name: wheel.read(name) for name in wheel.namelist()}
metadata = next(name for name in files if name.endswith(".dist-info/METADATA"))
record = next(name for name in files if name.endswith(".dist-info/RECORD"))
lines = files[metadata].decode().splitlines(keepends=True)
providers = [line for line in lines if line.startswith("Requires-Dist: flashinfer-jit-cache-sm")]
assert len(providers) >= 2, "Upstream cache dependency layout changed; review selection"
assert sum(line.startswith("Requires-Dist: " + selected + "==") for line in providers) == 1
files[metadata] = "".join(line for line in lines if line not in providers or
                          line.startswith("Requires-Dist: " + selected + "==")).encode()
manifest = next(name for name in files if name.endswith("/_provider_requirements.txt"))
selected_lines = [line for line in files[manifest].decode().splitlines()
                  if line.startswith(selected + "==")]
assert len(selected_lines) == 1, "Upstream provider manifest changed"
files[manifest] = (selected_lines[0] + "\n").encode()
rows = []
for name, data in files.items():
    if name != record:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append((name, "sha256=" + digest, len(data)))
rows.append((record, "", ""))
output = io.StringIO(newline="")
csv.writer(output).writerows(rows)
files[record] = output.getvalue().encode()
destination.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(destination, "w") as wheel:
    for name, data in files.items():
        wheel.writestr(name, data)
print(f"Kept upstream shim code and selected only {selected}")
