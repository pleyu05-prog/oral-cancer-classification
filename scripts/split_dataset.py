# scripts/split_dataset.py
import argparse, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def collect_images(p: Path):
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]

def split_one_class(files, ratios, seed=42):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源数据根目录（包含 CANCER/NON CANCER）")
    ap.add_argument("--dst", required=True, help="目标目录（将创建 train/val/test）")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert abs(args.train + args.val - 0.9) < 1e-6 and abs(args.train + args.val + args.test - 1.0) < 1e-6, "比例应和为1.0"

    classes = [d.name for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit("未在源目录下发现类别子文件夹。请确认 src 里有 CANCER / NON CANCER 等子目录。")

    for split in ["train","val","test"]:
        for c in classes:
            (dst / split / c).mkdir(parents=True, exist_ok=True)

    total_counts = {"train":0,"val":0,"test":0}
    for c in classes:
        files = collect_images(src / c)
        tr, va, te = split_one_class(files, (args.train, args.val, args.test), args.seed)
        for f in tr: shutil.copy2(f, dst / "train" / c / f.name)
        for f in va: shutil.copy2(f, dst / "val"   / c / f.name)
        for f in te: shutil.copy2(f, dst / "test"  / c / f.name)
        total_counts["train"] += len(tr)
        total_counts["val"]   += len(va)
        total_counts["test"]  += len(te)
        print(f"{c}: train={len(tr)} val={len(va)} test={len(te)}")

    print("DONE:", total_counts)

if __name__ == "__main__":
    main()
