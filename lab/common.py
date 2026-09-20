"""Small shared helpers for local, reproducible experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_texts(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: expected nonempty text")
            yield text


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_texts(path, texts):
    with open(path, "w", encoding="utf-8") as stream:
        for text in texts:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
