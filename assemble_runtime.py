"""Assemble only native runtime dependencies, preserving Distroless core ABI.

Ubuntu 24.04's CPython 3.12/extensions require glibc <= Debian 13's glibc.
Copy ELF dependency closure for Python, audio tools, GCC and extensions, but
never replace Distroless's glibc, dynamic loader, libgcc or libstdc++.
"""
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig

ROOT = Path("/runtime-root")
ROOT.mkdir()
core = re.compile(r"^(?:ld-linux.*|lib(?:c|m|dl|pthread|rt|resolv|gcc_s|stdc\+\+)\.so.*)$")
scanned = set()

def copy(path):
    src = Path(path)
    if not src.exists() or core.match(src.name):
        return
    destination = str(src)
    # Google Debian 13 is merged-/usr. Copying real top-level directories over
    # its /bin, /sbin, /lib or /lib64 symlinks fails before tests can execute.
    for prefix in ("/bin/", "/sbin/", "/lib/", "/lib64/"):
        if destination.startswith(prefix):
            destination = "/usr" + destination
            break
    dst = ROOT / destination.lstrip("/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)

def dependencies(path):
    src = Path(path)
    if str(src) in scanned or not src.is_file():
        return
    scanned.add(str(src))
    with src.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            return
    result = subprocess.run(["ldd", str(src)], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    for name in re.findall(r"(?:=>\s+|^\s+)(/[^\s]+)", result.stdout, re.MULTILINE):
        # CUDA runtime lives in the already-copied wheel environment, never in
        # the CUDA 13.4 builder toolkit. Avoid duplicating large venv libraries.
        if not name.startswith(("/usr/local/cuda", "/opt/venv/")):
            copy(name)

# Python sysconfig and extension compilers rely on these conventional paths.
copy("/usr/lib/python3.12")
copy("/usr/include")
copy("/usr/lib/gcc")
copy("/usr/share/sox")
for pattern in ["/usr/lib/*/libpython3.12.so*", "/usr/lib/*/libsox.so*",
                "/usr/lib/*/sox", "/usr/lib/*/libgomp.so*", "/usr/lib/*/libgcc_s.so",
                "/usr/lib/*/libstdc++.so", "/usr/lib/*/crt*.o"]:
    for path in glob.glob(pattern):
        copy(path)

tools = ["python3.12", "gcc", "g++", "cc", "c++", "as", "ld", "ar", "ranlib",
         "nm", "objdump", "objcopy", "strip", "readelf", "ffmpeg", "ffprobe",
         "sox", "env", "uname"]
for name in tools:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Missing required runtime tool: {name}")
    # Preserve real names too: GCC selects cc1 by executable installation path.
    copy(path)
    copy(str(Path(path).resolve()))
    dependencies(path)
copy("/bin/sh")  # Ninja/CUDA extension fallback invokes /bin/sh, not Bash.

for tree in ["/usr/lib/python3.12", "/usr/lib/gcc", "/opt/venv"]:
    for path in Path(tree).rglob("*"):
        if path.is_file() and (".so" in path.name or os.access(path, os.X_OK)):
            dependencies(path)
# FFmpeg and SoX use dlopen plugins, so scan those copied modules too.
for path in list(ROOT.rglob("*.so*")):
    original = Path("/") / path.relative_to(ROOT)
    dependencies(original)

# Link-time development symlinks refer to Distroless's newer ABI libraries.
multiarch = sysconfig.get_config_var("MULTIARCH")
libdir = ROOT / "usr/lib" / multiarch
libdir.mkdir(parents=True, exist_ok=True)
for short, soname in [("libc.so", "libc.so.6"), ("libm.so", "libm.so.6"),
                      ("libdl.so", "libdl.so.2"), ("libpthread.so", "libpthread.so.0"),
                      ("librt.so", "librt.so.1"), ("libgcc_s.so", "libgcc_s.so.1"),
                      ("libstdc++.so", "libstdc++.so.6")]:
    target = libdir / short
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(soname)
copy("/usr/share/doc/python3.12/copyright")
home = Path("/runtime-home/.cache")
home.mkdir(parents=True)
print(f"Runtime dependency closure assembled; {len(scanned)} ELF candidates inspected")
