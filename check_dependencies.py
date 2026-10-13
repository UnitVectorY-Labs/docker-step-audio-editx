"""Reject dependency inconsistencies, except NVIDIA's known ARM SBSA wheel tag."""
import platform
import subprocess
import sys

check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(check.stdout)
issues = check.stdout.strip().splitlines()
known_tag_issue = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
if check.returncode and not (platform.machine() == "aarch64" and issues == [known_tag_issue]):
    raise SystemExit(check.returncode)
if check.returncode:
    print("Accepted NVIDIA py3-none-manylinux2014_sbsa metadata tag on aarch64; "
          "the final runtime still checks real Torch imports. No dependency conflicts ignored.")
