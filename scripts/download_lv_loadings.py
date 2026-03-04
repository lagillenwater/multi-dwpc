#!/usr/bin/env python3
"""
Download LV loading files from a GitHub repository directory.

Default source:
- repo: greenelab/phenoplier
- path: data/input/multiplier
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "multi-dwpc-lv-loadings-downloader"


def _normalized_source_path(path: str) -> str:
    return "/".join(part for part in path.strip("/").split("/") if part)


def _contents_url(owner: str, repo: str, source_path: str, ref: str) -> str:
    encoded_path = "/".join(quote(part) for part in _normalized_source_path(source_path).split("/"))
    return (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{encoded_path}"
        f"?ref={quote(ref)}"
    )


def _http_json(url: str, token: str | None) -> dict | list:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def _download_bytes(url: str, token: str | None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download error {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def _is_git_lfs_pointer(payload: bytes) -> bool:
    return payload.startswith(b"version https://git-lfs.github.com/spec/v1")


def _media_download_url(owner: str, repo: str, ref: str, src_path: str) -> str:
    encoded_path = quote(src_path, safe="/")
    return f"https://media.githubusercontent.com/media/{owner}/{repo}/{quote(ref)}/{encoded_path}"


def _iter_files_recursive(
    owner: str,
    repo: str,
    source_path: str,
    ref: str,
    token: str | None,
) -> Iterable[dict]:
    payload = _http_json(_contents_url(owner, repo, source_path, ref), token)
    if isinstance(payload, dict):
        if payload.get("type") == "file":
            yield payload
        elif payload.get("type") == "dir":
            yield from _iter_files_recursive(
                owner=owner,
                repo=repo,
                source_path=payload["path"],
                ref=ref,
                token=token,
            )
        return

    for item in payload:
        item_type = item.get("type")
        if item_type == "file":
            yield item
        elif item_type == "dir":
            yield from _iter_files_recursive(
                owner=owner,
                repo=repo,
                source_path=item["path"],
                ref=ref,
                token=token,
            )


def _matches_any_pattern(path: str, patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    return any(pattern.search(path) for pattern in patterns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LV loading files from a GitHub directory."
    )
    parser.add_argument("--owner", default="greenelab", help="GitHub owner/org.")
    parser.add_argument("--repo", default="phenoplier", help="GitHub repository.")
    parser.add_argument(
        "--source-path",
        default="data/input/multiplier",
        help="Directory path in the source repository.",
    )
    parser.add_argument("--ref", default="main", help="Git ref (branch/tag/sha).")
    parser.add_argument(
        "--output-dir",
        default="data/lv_loadings",
        help="Local output directory for downloaded files.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Regex to filter files by source path; can be repeated. "
            "If omitted, downloads all files under --source-path."
        ),
    )
    parser.add_argument(
        "--exact-file",
        action="append",
        default=[],
        help=(
            "Exact filename(s) to download from the source directory. "
            "Can be repeated. Example: --exact-file multiplier_model_z.tsv.gz"
        ),
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing optional GitHub token.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv(args.token_env)
    output_dir = Path(args.output_dir).resolve()
    source_root = _normalized_source_path(args.source_path)
    patterns = [re.compile(expr) for expr in args.pattern]
    exact_files = set(args.exact_file)

    print(
        f"Listing files from https://github.com/{args.owner}/{args.repo}/tree/"
        f"{args.ref}/{source_root}"
    )
    files = list(
        _iter_files_recursive(
            owner=args.owner,
            repo=args.repo,
            source_path=source_root,
            ref=args.ref,
            token=token,
        )
    )
    if not files:
        raise RuntimeError(
            "No files found at source path. Check --owner/--repo/--source-path/--ref."
        )

    selected = []
    for item in files:
        src_path = item.get("path", "")
        src_name = Path(src_path).name
        if exact_files and src_name not in exact_files:
            continue
        if not _matches_any_pattern(src_path, patterns):
            continue
        selected.append(item)
    if not selected:
        raise RuntimeError("No files matched the provided --pattern filters.")

    print(f"Found {len(files)} files; selected {len(selected)} for download.")
    for item in selected:
        src_path = item["path"]
        rel_path = Path(src_path).relative_to(source_root)
        dest_path = output_dir / rel_path
        print(f"  {src_path} -> {dest_path}")

    if args.dry_run:
        print("Dry run complete; no files downloaded.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for item in selected:
        src_path = item["path"]
        rel_path = Path(src_path).relative_to(source_root)
        dest_path = output_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        download_url = item.get("download_url")
        if not download_url:
            raise RuntimeError(f"Missing download_url for {src_path}")

        payload = _download_bytes(download_url, token)
        if _is_git_lfs_pointer(payload):
            media_url = _media_download_url(args.owner, args.repo, args.ref, src_path)
            payload = _download_bytes(media_url, token)

        if _is_git_lfs_pointer(payload):
            raise RuntimeError(
                f"Downloaded {src_path} but received a Git LFS pointer instead of file content."
            )

        dest_path.write_bytes(payload)
        downloaded += 1

    print(f"Download complete. Wrote {downloaded} files to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
