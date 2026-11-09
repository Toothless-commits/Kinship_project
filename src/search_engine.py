import argparse
import sys
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

# Ensure src/ directory is always in sys.path for direct runs and root imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from model import TriSubjectKinshipNet
except ImportError:
    from src.model import TriSubjectKinshipNet


class ParentSearchEngine:
    def __init__(
        self,
        weights_path: Path = Path("weights/best_model.pt"),
        triplets_csv: Path = Path("data/triplets.csv"),
        aligned_dir: Path = Path("data/aligned"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        print(f"[+] Initializing Parent Search Engine on {self.device}...")

        # 1. Face alignment and preprocessing
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            post_process=False,
            select_largest=True,
            device=self.device,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # 2. Load neural network weights
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

        self.model = TriSubjectKinshipNet(freeze_backbone=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 3. Build parent candidate gallery
        self.triplets_csv = Path(triplets_csv)
        self.aligned_dir = Path(aligned_dir)
        self.gallery_df, self.f_emb, self.m_emb = self._index_parent_gallery()

    def _load_and_transform(self, path: Path) -> torch.Tensor:
        with Image.open(path).convert("RGB") as img:
            return self.transform(img)

    def _index_parent_gallery(self):
        """Indexes all unique parent couples and precomputes their 512-d embeddings."""
        if not self.triplets_csv.exists():
            raise FileNotFoundError(f"Triplets CSV not found at: {self.triplets_csv}")

        df = pd.read_csv(self.triplets_csv)

        def to_aligned(raw_p: str) -> str:
            parts = Path(raw_p).parts
            f_idx = next(
                i
                for i, part in enumerate(parts)
                if part.startswith("F") and len(part) >= 4 and part[1:].isdigit()
            )
            return str(self.aligned_dir / Path(*parts[f_idx:]))

        df["father_path"] = df["father_path"].apply(to_aligned)
        df["mother_path"] = df["mother_path"].apply(to_aligned)

        valid_mask = df["father_path"].apply(lambda p: Path(p).is_file()) & df[
            "mother_path"
        ].apply(lambda p: Path(p).is_file())
        gallery_df = (
            df[valid_mask][["family_id", "father_path", "mother_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(
            f"[+] Precomputing embeddings for {len(gallery_df)} candidate parent couples..."
        )

        f_tensors = [
            self._load_and_transform(Path(p)) for p in gallery_df["father_path"]
        ]
        m_tensors = [
            self._load_and_transform(Path(p)) for p in gallery_df["mother_path"]
        ]

        f_stack = torch.stack(f_tensors).to(self.device)
        m_stack = torch.stack(m_tensors).to(self.device)

        with torch.no_grad():
            f_emb = torch.cat(
                [self.model.extract_features(b) for b in torch.split(f_stack, 32)],
                dim=0,
            )
            m_emb = torch.cat(
                [self.model.extract_features(b) for b in torch.split(m_stack, 32)],
                dim=0,
            )

        print(f"[+] Indexing ready: {f_emb.shape[0]} parent pairs cached in VRAM.")
        return gallery_df, f_emb, m_emb

    def extract_child_tensor(
        self, img_input: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """Detects, aligns, and standardizes a child input into a [1, 3, 160, 160] tensor."""
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input).convert("RGB")
        else:
            img = img_input.convert("RGB")

        face_tensor = self.mtcnn(img)
        if face_tensor is not None:
            face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().numpy())
        else:
            # Fallback center crop if detection threshold is low
            w, h = img.size
            min_dim = min(w, h)
            face_img = img.crop(
                (
                    (w - min_dim) // 2,
                    (h - min_dim) // 2,
                    (w + min_dim) // 2,
                    (h + min_dim) // 2,
                )
            ).resize((160, 160))

        return self.transform(face_img).unsqueeze(0).to(self.device)

    def search(
        self, child_input: Union[str, Path, Image.Image], top_k: int = 5
    ) -> List[Dict]:
        """
        Executes vectorized 1-to-N kinship matching against the cached parent gallery.
        """
        child_tensor = self.extract_child_tensor(child_input)

        with torch.no_grad():
            c_emb = self.model.extract_features(child_tensor)  # [1, 512]
            n_parents = self.f_emb.shape[0]
            c_expanded = c_emb.expand(n_parents, -1)  # [N, 512]

            # Bilinear Relational Fusion
            diff_fc = torch.abs(self.f_emb - c_expanded)
            diff_mc = torch.abs(self.m_emb - c_expanded)
            prod_fc = self.f_emb * c_expanded
            prod_mc = self.m_emb * c_expanded
            mid_parent = (self.f_emb + self.m_emb) / 2.0
            diff_midc = torch.abs(mid_parent - c_expanded)

            fused = torch.cat(
                [diff_fc, diff_mc, prod_fc, prod_mc, diff_midc], dim=1
            )  # [N, 2560]

            logits = self.model.classifier(fused).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

        ranked_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.gallery_df.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "family_id": row["family_id"],
                    "father_path": row["father_path"],
                    "mother_path": row["mother_path"],
                }
            )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 1-to-N parent search for a child image"
    )
    parser.add_argument(
        "--child", type=Path, help="Path to child face image", required=False
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of ranked parents to retrieve"
    )
    args = parser.parse_args()

    engine = ParentSearchEngine()

    if args.child is None:
        df = pd.read_csv("data/triplets.csv")
        test_child = Path(df.iloc[0]["child_path"])
        parts = test_child.parts
        f_idx = next(
            i
            for i, part in enumerate(parts)
            if part.startswith("F") and part[1:].isdigit()
        )
        sample_path = Path("data/aligned") / Path(*parts[f_idx:])
        print(f"\n[+] Testing with sample child image: {sample_path}")
        target = sample_path
    else:
        target = args.child

    matches = engine.search(target, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"           TOP-{args.top_k} PARENT MATCHES FOUND")
    print("=" * 60)
    for m in matches:
        print(
            f"Rank {m['rank']} | Confidence: {m['score'] * 100:.2f}% | Family: {m['family_id']}"
        )
        print(f"       Father: {m['father_path']}")
        print(f"       Mother: {m['mother_path']}")
        print("-" * 60)
