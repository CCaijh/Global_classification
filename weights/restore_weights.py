#!/usr/bin/env python3
"""Verify release assets and restore the original model weight files."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

CHUNK_SIZE = 8 * 1024 * 1024


def safe_relative(value, label, single_name=False):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Invalid {label}: {value!r}")
    if single_name and len(path.parts) != 1:
        raise ValueError(f"Invalid asset name: {value!r}")
    return Path(*path.parts)


def metadata(item):
    size = item.get("size")
    digest = item.get("sha256")
    if type(size) is not int or size < 0 or not isinstance(digest, str):
        raise ValueError("Invalid size or SHA256 in manifest")
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("Invalid SHA256 in manifest")
    return size, digest.lower()


def within(root, relative):
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise ValueError(f"Path leaves the selected directory: {relative}") from None
    return path


def file_digest(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            count += len(chunk)
    return count, digest.hexdigest()


def restore(manifest_path, assets_dir, output_dir):
    assets_dir = assets_dir.resolve()
    output_dir = output_dir.resolve()
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain a nonempty files list")

    planned = []
    seen = set()
    for item in files:
        relative = safe_relative(item["source_relative_path"], "output path")
        identity = str(relative).casefold()
        if identity in seen:
            raise ValueError(f"Duplicate output path: {relative}")
        seen.add(identity)
        expected = metadata(item)
        parts = item.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"Missing parts for {relative}")
        checked_parts = []
        for part in parts:
            part_path = safe_relative(part["name"], "asset name", single_name=True)
            checked_parts.append((within(assets_dir, part_path), metadata(part)))
        if sum(spec[0] for _, spec in checked_parts) != expected[0]:
            raise ValueError(f"Part sizes do not match file size: {relative}")
        planned.append((relative, expected, checked_parts))

    for relative, expected, parts in planned:
        target = within(output_dir, relative)
        if os.path.lexists(target):
            if target.is_file() and file_digest(target) == expected:
                print(f"Already verified: {relative}", flush=True)
                continue
            raise FileExistsError(f"Refusing to overwrite an existing file: {target}")

        # Check every source part before creating the restored file.
        for asset, specification in parts:
            print(f"Verifying: {asset.name}", flush=True)
            if file_digest(asset) != specification:
                raise ValueError(f"Asset size or SHA256 mismatch: {asset.name}")

        target.parent.mkdir(parents=True, exist_ok=True)
        within(output_dir, relative)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".tmp",
                dir=target.parent, delete=False,
            ) as output:
                temporary_path = Path(output.name)
                restored_hash = hashlib.sha256()
                restored_size = 0
                for asset, specification in parts:
                    part_hash = hashlib.sha256()
                    part_size = 0
                    with asset.open("rb") as source:
                        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                            output.write(chunk)
                            restored_hash.update(chunk)
                            restored_size += len(chunk)
                            part_hash.update(chunk)
                            part_size += len(chunk)
                    if (part_size, part_hash.hexdigest()) != specification:
                        raise ValueError(f"Asset changed while restoring: {asset.name}")
            if (restored_size, restored_hash.hexdigest()) != expected:
                raise ValueError(f"Restored file size or SHA256 mismatch: {relative}")
            # Windows rename refuses an existing target. On POSIX, hard-linking
            # gives the same no-overwrite guarantee, then removes the temp name.
            if os.name == "nt":
                os.rename(temporary_path, target)
            else:
                os.link(temporary_path, target)
                temporary_path.unlink()
            temporary_path = None
            print(f"Restored and verified: {relative}", flush=True)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=directory / "manifest.json")
    parser.add_argument("--assets", type=Path, default=directory,
                        help="Directory containing all downloaded release assets")
    parser.add_argument("--output", type=Path, default=directory,
                        help="Directory in which to restore the four .pth files")
    args = parser.parse_args()
    try:
        restore(args.manifest, args.assets, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
