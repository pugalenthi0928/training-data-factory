#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_url


def main():
    ap = argparse.ArgumentParser(description="Publish files to a Hugging Face Dataset repo.")
    ap.add_argument("--repo", required=True, help="eg: yourname/training-data-factory-papers-qa")
    ap.add_argument("--path", action="append", required=True, help="Local file to upload (can repeat)")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_TOKEN env var not set.", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    try:
        create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    except Exception as e:
        print(
            f"[publish_hf] create_repo failed: {e}\nHint: ensure HUGGINGFACE_TOKEN has repo WRITE and the namespace matches your account.",
            flush=True,
        )
        raise

    for p in args.path:
        pth = Path(p)
        if not pth.exists():
            print(f"Skip missing: {pth}", file=sys.stderr)
            continue
        api.upload_file(
            path_or_fileobj=str(pth),
            path_in_repo=pth.name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"Uploaded {pth} → {hf_hub_url(args.repo, pth.name, repo_type='dataset')}")
    print("Publish complete.")


if __name__ == "__main__":
    main()
