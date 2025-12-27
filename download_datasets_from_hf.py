#!/usr/bin/env python3
"""
Download datasets from Hugging Face Hub for cloud training.
Requires HF_AUTH_TOKEN in environment.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables from .env file if it exists
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_AUTH_TOKEN")

if not HF_TOKEN:
    print("⚠ HF_AUTH_TOKEN not found")
    print("  Option 1: Create/update .env file with: HF_AUTH_TOKEN=hf_...")
    print("  Option 2: Export in shell: export HF_AUTH_TOKEN='hf_...'")
    exit(1)


def download_datasets():
    """Download datasets from HF Hub to local data/train/."""

    # Configuration
    repo_id = "teru00801/person-finetune-dataset1"
    repo_type = "dataset"
    files_to_download = [
        "hawks_val_segments.jsonl",
        "hawks_train_segments.jsonl",
        "person_dummy_segments.jsonl"
    ]

    base_dir = Path("/workspace/data/train")  # Cloud path (adjust if needed)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Hugging Face Dataset Download (Cloud)")
    print(f"{'='*80}")
    print(f"Repository: {repo_id}")
    print(f"Target dir: {base_dir}")
    print(f"Files: {len(files_to_download)}\n")

    all_valid = True
    total_size = 0

    for file_name in files_to_download:
        output_path = base_dir / file_name

        # Skip if already exists
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"  ✓ {file_name}: Already exists ({file_size / (1024*1024):.2f} MB)")
            total_size += file_size
            continue

        print(f"  Downloading {file_name}...", end="", flush=True)

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                repo_type=repo_type,
                cache_dir=str(base_dir.parent / ".cache"),
                force_download=False,
                resume_download=True
            )

            # Move from cache to target location
            import shutil
            shutil.move(downloaded_path, str(output_path))

            file_size = output_path.stat().st_size
            print(f" ✓ ({file_size / (1024*1024):.2f} MB)")
            total_size += file_size

        except Exception as e:
            print(f" ✗ Error: {str(e)[:60]}")
            all_valid = False

    if not all_valid:
        print("\n✗ Download failed. Check your HF_AUTH_TOKEN.")
        exit(1)

    # Verify downloads
    print(f"\n{'='*80}")
    print("Verification")
    print(f"{'='*80}")

    total_records = 0
    for file_name in files_to_download:
        file_path = base_dir / file_name
        if not file_path.exists():
            print(f"  ✗ {file_name}: File not found")
            continue

        try:
            record_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        record_count += 1

            print(f"  ✓ {file_name}: {record_count:,} records")
            total_records += record_count

        except json.JSONDecodeError as e:
            print(f"  ✗ {file_name}: JSON error at line {record_count}")

    print(f"\nTotal records: {total_records:,}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")

    print(f"\n{'='*80}")
    print("✓ Ready for training")
    print(f"{'='*80}")
    print(f"\nYou can now run:")
    print(f"  axolotl train src/axolotl_configs/rakuten_7b_phase1.yml")


if __name__ == "__main__":
    download_datasets()
