"""Fail on credentials, cloud storage paths, private endpoints or personal paths in publishable files."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "HF token": r"hf_[A-Za-z0-9]{24,}",
    "GitHub token": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "API secret": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "bucket path": r"\b(?:gs|r2|s3)://",
    "cloud project": r"gen-lang-client-\d+",
    "private endpoint": r"(?i)\.openai\.azure\.com|\.azure\.com",
    "local personal path": r"/Users/[A-Za-z]|/home/[A-Za-z]",
    "evaluation id": r"(?i)\beval(?: run| id)? [0-9a-f]{8}\b",
}


def main():
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                    cwd=ROOT).decode().split("\0")
    errors = []
    for name in sorted(set(filter(None, paths))):
        path = ROOT / name
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append((name, "environment file"))
        if name == "scripts/check_public_files.py" or not path.is_file():
            continue
        data = path.read_text(errors="replace")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, data):
                errors.append((name, label))
    if errors:
        raise SystemExit("Public-file review failed: " + repr(errors))
    print("Publishable files: no detected credentials or private infrastructure references")


if __name__ == "__main__":
    main()
