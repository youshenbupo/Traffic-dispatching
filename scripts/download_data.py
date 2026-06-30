"""Download public TSC datasets.

Currently supports:
- LibSignal cross-simulator benchmark (SUMO + CityFlow formats)
  https://github.com/DaRL-LibSignal/LibSignal
"""
import os
import sys
import argparse
import shutil
import subprocess


LIBSIGNAL_REPO = "https://github.com/DaRL-LibSignal/LibSignal.git"


def run_cmd(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def download_libsignal(out_dir, keep_git=False):
    """Download LibSignal data/raw_data via git sparse-checkout.

    This avoids the large full-repo download and is generally more reliable than
    the GitHub archive zip on flaky networks.
    """
    target = os.path.join(out_dir, "libsignal")
    git_dir = os.path.join(out_dir, ".libsignal_git")

    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)
    os.makedirs(git_dir, exist_ok=True)

    print("Cloning LibSignal with sparse-checkout for data/raw_data ...")
    run_cmd(["git", "init"], cwd=git_dir)
    run_cmd(["git", "remote", "add", "origin", LIBSIGNAL_REPO], cwd=git_dir)
    run_cmd(["git", "config", "core.sparseCheckout", "true"], cwd=git_dir)

    sparse_file = os.path.join(git_dir, ".git", "info", "sparse-checkout")
    with open(sparse_file, "w") as f:
        f.write("data/raw_data/*\n")

    run_cmd(["git", "pull", "--depth=1", "origin", "master"], cwd=git_dir)

    raw_data = os.path.join(git_dir, "data", "raw_data")
    if not os.path.isdir(raw_data):
        raise RuntimeError(f"Sparse checkout did not produce {raw_data}")

    os.makedirs(target, exist_ok=True)
    for name in os.listdir(raw_data):
        src = os.path.join(raw_data, name)
        dst = os.path.join(target, name)
        if os.path.exists(dst):
            print(f"Skipping {name} (already exists)")
            continue
        shutil.move(src, dst)
        print(f"Moved {name}")

    if keep_git:
        print(f"Keeping sparse git clone at {git_dir}")
    else:
        shutil.rmtree(git_dir)

    print(f"\nLibSignal datasets saved to {target}")
    print("Available scenarios:")
    for name in sorted(os.listdir(target)):
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="libsignal",
                        choices=["libsignal", "cityflow"])
    parser.add_argument("--out_dir", type=str, default="data/datasets")
    parser.add_argument("--keep-git", action="store_true",
                        help="Keep the intermediate sparse git clone.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "libsignal":
        download_libsignal(args.out_dir, keep_git=args.keep_git)
    elif args.dataset == "cityflow":
        print("CityFlow standalone download not implemented. "
              "Use --dataset libsignal to get both SUMO and CityFlow formats.")


if __name__ == "__main__":
    main()
