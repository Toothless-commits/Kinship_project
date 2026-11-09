import argparse
from pathlib import Path
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def align_and_cache_faces(
    csv_path: Path, output_dir: Path, img_size: int = 160, device: str = "cuda"
):
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique image paths across father, mother, and child columns
    all_raw_paths = (
        set(df["father_path"])
        .union(set(df["mother_path"]))
        .union(set(df["child_path"]))
    )
    unique_paths = sorted([Path(p) for p in all_raw_paths if Path(p).is_file()])

    print(f"[+] Total unique face images to process: {len(unique_paths)}")

    # Initialize MTCNN detector
    mtcnn = MTCNN(
        image_size=img_size,
        margin=20,
        post_process=False,  # Keep outputs as raw uint8 tensors
        select_largest=True,
        device=device,
    )

    failed_crops = 0

    for raw_path in tqdm(unique_paths, desc="Aligning faces"):
        # Create a matching path inside data/aligned/<family_id>/<mid>/<img_name>
        # raw_path looks like: .../train-faces/F0016/MID2/P00162_face2.jpg
        parts = raw_path.parts
        try:
            # Locate family folder (e.g. F0016)
            f_idx = next(
                i
                for i, part in enumerate(parts)
                if part.startswith("F") and len(part) >= 4 and part[1:].isdigit()
            )
            rel_path = Path(*parts[f_idx:])
        except StopIteration:
            rel_path = Path(raw_path.parent.name) / raw_path.name

        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        try:
            with Image.open(raw_path).convert("RGB") as img:
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    face_img = Image.fromarray(
                        face_tensor.permute(1, 2, 0).byte().numpy()
                    )
                    face_img.save(out_path, quality=95)
                else:
                    # Fallback center crop if face detection confidence is low
                    w, h = img.size
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    fallback_crop = img.crop(
                        (left, top, left + min_dim, top + min_dim)
                    ).resize((img_size, img_size))
                    fallback_crop.save(out_path, quality=95)
                    failed_crops += 1
        except Exception:
            failed_crops += 1

    print(f"\n[+] Alignment complete. Saved crops to: {output_dir}")
    print(f"[!] Fallback center crops used: {failed_crops}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and align faces using MTCNN")
    parser.add_argument("--csv", type=Path, default=Path("data/triplets.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/aligned"))
    parser.add_argument("--img_size", type=int, default=160)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running MTCNN on: {dev}")
    align_and_cache_faces(args.csv, args.out_dir, img_size=args.img_size, device=dev)
