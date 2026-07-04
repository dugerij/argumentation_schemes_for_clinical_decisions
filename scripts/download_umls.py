from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Any

import requests


DEFAULT_RELEASE_API = "https://uts-ws.nlm.nih.gov/releases"
DEFAULT_DOWNLOAD_API = "https://uts-ws.nlm.nih.gov/download"
DEFAULT_RELEASE_TYPE = "umls-metathesaurus-full-subset"
DEFAULT_OUTPUT_DIR = Path("data/umls")


def _find_first_download_url(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for value in payload.values():
            match = _find_first_download_url(value)
            if match:
                return match
        return None
    if isinstance(payload, list):
        for item in payload:
            match = _find_first_download_url(item)
            if match:
                return match
        return None
    if isinstance(payload, str) and payload.startswith("https://download.nlm.nih.gov/") and payload.endswith(".zip"):
        return payload
    return None


def resolve_release_download_url(
    *,
    release_type: str,
    current: bool,
    release_api: str = DEFAULT_RELEASE_API,
    timeout: float = 60.0,
) -> str:
    response = requests.get(
        release_api,
        params={"releaseType": release_type, "current": str(current).lower()},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    download_url = _find_first_download_url(payload)
    if not download_url:
        raise ValueError(
            "Could not find a release download URL in the Release API response. "
            f"releaseType={release_type}"
        )
    return download_url


def download_via_uts(
    *,
    api_key: str,
    release_download_url: str,
    destination: Path,
    download_api: str = DEFAULT_DOWNLOAD_API,
    timeout: float = 600.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {"url": release_download_url, "apiKey": api_key}
    response = requests.get(download_api, params=params, stream=True, timeout=timeout)
    response.raise_for_status()

    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return destination


def extract_umls_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    return output_dir


def find_meta_dir(root: Path) -> Path | None:
    direct = root / "META"
    if direct.exists():
        return direct
    for candidate in root.rglob("META"):
        if candidate.is_dir() and (candidate / "MRCONSO.RRF").exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a UMLS release through the UTS Release/Download APIs.")
    parser.add_argument("--api-key", default=os.environ.get("UMLS_API_KEY"))
    parser.add_argument("--release-type", default=DEFAULT_RELEASE_TYPE)
    parser.add_argument("--release-url", default=None, help="Optional direct NLM release zip URL. Skips Release API discovery.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--zip-name", default=None, help="Optional zip filename to write under output-dir.")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--current", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Set UMLS_API_KEY or pass --api-key.")

    output_dir = Path(args.output_dir)
    release_url = args.release_url or resolve_release_download_url(
        release_type=args.release_type,
        current=args.current,
    )

    zip_name = args.zip_name or Path(release_url).name
    zip_path = output_dir / zip_name

    print(f"Resolved release URL: {release_url}")
    print(f"Downloading via UTS Download API to {zip_path}")
    download_via_uts(
        api_key=args.api_key,
        release_download_url=release_url,
        destination=zip_path,
    )

    if args.no_extract:
        print(f"Downloaded {zip_path}")
        return

    print(f"Extracting {zip_path} into {output_dir}")
    extract_umls_zip(zip_path, output_dir)
    meta_dir = find_meta_dir(output_dir)
    if meta_dir:
        print(f"Found META directory: {meta_dir}")
    else:
        print("Extraction completed, but META directory was not found automatically.")

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"Removed archive {zip_path}")


if __name__ == "__main__":
    main()
