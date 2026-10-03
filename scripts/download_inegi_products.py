"""Download INEGI source products declared in data/sources/inegi_products.json.

Large binary archives stay out of Git history. The manifest records authoritative
source URLs and SHA-256 checksums so experiments remain reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path("data/sources/inegi_products.json")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "JEPA-Green-Audit/1.0"},
    )

    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = 100.0 * downloaded / total_bytes
                print(
                    f"  {downloaded / (1024**2):.1f} / "
                    f"{total_bytes / (1024**2):.1f} MiB ({pct:.1f}%)",
                    end="\r",
                )

    if total_bytes:
        print()
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even when a valid local file already exists.",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    products = payload.get("products", [])
    if not products:
        raise ValueError("Manifest contains no products")

    for product in products:
        filename = product["filename"]
        destination = Path(product.get("target_dir", "data/raw/inegi")) / filename
        expected_raw = product.get("sha256")
        expected = expected_raw.lower() if expected_raw else None
        url = product["source_url"]

        print(f"{product['sheet']} — {product['product']}")
        print(f"  destination: {destination}")

        if destination.exists() and not args.force:
            actual = sha256_file(destination)
            if expected and actual == expected:
                print("  already present; SHA-256 verified")
                continue
            if not expected:
                print(f"  already present; SHA-256 computed: {actual}")
                print("  manifest checksum is pending verification")
                continue
            print("  existing file checksum mismatch; redownloading")

        print(f"  downloading: {url}")
        download_file(url, destination)

        actual = sha256_file(destination)
        if expected:
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
                )
            print(f"  SHA-256 verified: {actual}")
        else:
            print(f"  SHA-256 computed: {actual}")
            print("  NOTE: add this checksum to the manifest to pin the binary exactly.")

    print("All INEGI products are ready.")


if __name__ == "__main__":
    main()
