import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from model import TriSubjectKinshipNet
except ImportError:
    from src.model import TriSubjectKinshipNet


GALLERY_COLUMNS = [
    "family_id",
    "father_name",
    "mother_name",
    "father_path",
    "mother_path",
    "source",
]


def _split_paths(path_field: str) -> List[Path]:
    return [Path(p) for p in str(path_field).split("|") if p.strip()]


class ParentSearchEngine:
    def __init__(
        self,
        weights_path: Path = Path("weights/best_model.pt"),
        triplets_csv: Path = Path("data/triplets.csv"),
        aligned_dir: Path = Path("data/aligned"),
        custom_gallery_dir: Path = Path("data/custom_gallery"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        print(f"[+] Initializing Parent Search Engine on {self.device}...")

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

        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

        self.model = TriSubjectKinshipNet(freeze_backbone=False)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        self.triplets_csv = Path(triplets_csv)
        self.aligned_dir = Path(aligned_dir)
        self.custom_gallery_dir = Path(custom_gallery_dir)
        self.custom_gallery_dir.mkdir(parents=True, exist_ok=True)
        self.custom_csv = self.custom_gallery_dir / "gallery.csv"

        self.gallery_df, self.f_emb, self.m_emb = self._index_parent_gallery()

    def _load_and_transform(self, path: Path) -> torch.Tensor:
        with Image.open(path).convert("RGB") as img:
            return self.transform(img)

    @torch.no_grad()
    def _extract_tta_batch(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        emb_orig = self.model.extract_features(tensor_batch)
        emb_flip = self.model.extract_features(torch.flip(tensor_batch, dims=[3]))
        return F.normalize(emb_orig + emb_flip, p=2, dim=1)

    @torch.no_grad()
    def _embed_path_field(self, path_field: str) -> torch.Tensor:
        """Computes an L2-normalized TTA prototype across 1 or more aligned photos."""
        paths = [p for p in _split_paths(path_field) if p.is_file()]
        tensors = torch.stack([self._load_and_transform(p) for p in paths]).to(
            self.device
        )
        embs = self._extract_tta_batch(tensors)
        return F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1)

    def _load_fiw_pairs(self) -> pd.DataFrame:
        if not self.triplets_csv.exists():
            return pd.DataFrame(columns=GALLERY_COLUMNS)

        df = pd.read_csv(self.triplets_csv)

        def to_aligned(raw_p: str) -> str:
            parts = Path(raw_p).parts
            try:
                f_idx = next(
                    i
                    for i, part in enumerate(parts)
                    if part.startswith("F") and len(part) >= 4 and part[1:].isdigit()
                )
                return str(self.aligned_dir / Path(*parts[f_idx:]))
            except StopIteration:
                return str(raw_p)

        df["father_path"] = df["father_path"].apply(to_aligned)
        df["mother_path"] = df["mother_path"].apply(to_aligned)

        valid_mask = df["father_path"].apply(lambda p: Path(p).is_file()) & df[
            "mother_path"
        ].apply(lambda p: Path(p).is_file())

        fiw_df = (
            df[valid_mask][["family_id", "father_path", "mother_path"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        fiw_df["father_name"] = ""
        fiw_df["mother_name"] = ""
        fiw_df["source"] = "FIW Dataset"
        return fiw_df[GALLERY_COLUMNS]

    def _load_custom_pairs(self) -> pd.DataFrame:
        if not self.custom_csv.exists():
            return pd.DataFrame(columns=GALLERY_COLUMNS)

        df = pd.read_csv(self.custom_csv).fillna("")
        for col in GALLERY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df["source"] = "Custom"

        valid_mask = df["father_path"].apply(
            lambda pf: any(p.is_file() for p in _split_paths(pf))
        ) & df["mother_path"].apply(
            lambda pf: any(p.is_file() for p in _split_paths(pf))
        )
        return df[valid_mask][GALLERY_COLUMNS].reset_index(drop=True)

    def _index_parent_gallery(self) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        custom_df = self._load_custom_pairs()
        fiw_df = self._load_fiw_pairs()

        gallery_df = (
            pd.concat([custom_df, fiw_df], ignore_index=True)
            .drop_duplicates(subset=["father_path", "mother_path"])
            .reset_index(drop=True)
        )

        if gallery_df.empty:
            print("[!] Gallery is currently empty. Add parents via Gallery Manager.")
            empty_emb = torch.empty((0, 512), device=self.device)
            return pd.DataFrame(columns=GALLERY_COLUMNS), empty_emb, empty_emb

        print(
            f"[+] Precomputing TTA embeddings for {len(gallery_df)} parent couples "
            f"({len(custom_df)} custom, {len(fiw_df)} dataset)..."
        )

        f_vecs, m_vecs = [], []

        # Custom couples may contain multiple photos per parent (prototype averaging)
        if not custom_df.empty:
            for _, row in custom_df.iterrows():
                f_vecs.append(self._embed_path_field(row["father_path"]))
                m_vecs.append(self._embed_path_field(row["mother_path"]))

        # Dataset couples are batched on GPU for fast startup
        fiw_only = gallery_df[gallery_df["source"] == "FIW Dataset"]
        if not fiw_only.empty:
            f_stack = torch.stack(
                [self._load_and_transform(Path(p)) for p in fiw_only["father_path"]]
            ).to(self.device)
            m_stack = torch.stack(
                [self._load_and_transform(Path(p)) for p in fiw_only["mother_path"]]
            ).to(self.device)

            with torch.no_grad():
                f_fiw = torch.cat(
                    [self._extract_tta_batch(b) for b in torch.split(f_stack, 32)],
                    dim=0,
                )
                m_fiw = torch.cat(
                    [self._extract_tta_batch(b) for b in torch.split(m_stack, 32)],
                    dim=0,
                )
            f_vecs.append(f_fiw)
            m_vecs.append(m_fiw)

        f_emb = torch.cat(f_vecs, dim=0)
        m_emb = torch.cat(m_vecs, dim=0)

        print(f"[+] Indexing ready: {f_emb.shape[0]} parent pairs cached in VRAM.")
        return gallery_df, f_emb, m_emb

    def align_face_image(
        self, img_input: Union[str, Path, Image.Image]
    ) -> Tuple[Image.Image, bool]:
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input).convert("RGB")
        else:
            img = img_input.convert("RGB")

        face_tensor = self.mtcnn(img)
        if face_tensor is not None:
            face_img = Image.fromarray(face_tensor.permute(1, 2, 0).byte().numpy())
            return face_img, True

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
        return face_img, False

    def extract_child_tensor(
        self, img_input: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        face_img, _ = self.align_face_image(img_input)
        return self.transform(face_img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_tta_embedding(
        self, img_input: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        t = self.extract_child_tensor(img_input)
        return self._extract_tta_batch(t)

    @torch.no_grad()
    def verify_triplet(
        self,
        img_f: Union[str, Path, Image.Image],
        img_m: Union[str, Path, Image.Image],
        img_c: Union[str, Path, Image.Image],
    ) -> float:
        """Runs TTA-boosted 1-to-1 biological triad verification."""
        ef = self.extract_tta_embedding(img_f)
        em = self.extract_tta_embedding(img_m)
        ec = self.extract_tta_embedding(img_c)
        logit = self.model.forward_from_embeddings(ef, em, ec)
        return float(torch.sigmoid(logit).item())

    def next_custom_family_id(self) -> str:
        custom_df = self.get_custom_gallery()
        existing = set(custom_df["family_id"].tolist())
        idx = 1
        while f"FAM_{idx:03d}" in existing:
            idx += 1
        return f"FAM_{idx:03d}"

    def get_custom_gallery(self) -> pd.DataFrame:
        if self.gallery_df.empty:
            return pd.DataFrame(columns=GALLERY_COLUMNS)
        return self.gallery_df[self.gallery_df["source"] == "Custom"].reset_index(
            drop=True
        )

    def add_parent_couple(
        self,
        family_id: str,
        father_imgs: Union[Image.Image, Sequence[Image.Image]],
        mother_imgs: Union[Image.Image, Sequence[Image.Image]],
        father_name: str = "",
        mother_name: str = "",
    ) -> Dict:
        """
        Aligns 1 or more photos per parent, saves them to data/custom_gallery/,
        builds averaged TTA prototypes, and updates the live VRAM index.
        """
        clean_id = re.sub(r"[^A-Za-z0-9_-]", "_", family_id.strip())
        if not clean_id:
            clean_id = self.next_custom_family_id()

        f_list = (
            [father_imgs]
            if isinstance(father_imgs, (str, Path, Image.Image))
            else list(father_imgs)
        )
        m_list = (
            [mother_imgs]
            if isinstance(mother_imgs, (str, Path, Image.Image))
            else list(mother_imgs)
        )

        pair_dir = self.custom_gallery_dir / clean_id
        if pair_dir.exists():
            shutil.rmtree(pair_dir, ignore_errors=True)
        pair_dir.mkdir(parents=True, exist_ok=True)

        f_paths, f_aligned_list, f_all_detected = [], [], True
        for i, img in enumerate(f_list):
            aligned, ok = self.align_face_image(img)
            out_p = pair_dir / f"father_{i}.jpg"
            aligned.save(out_p, quality=95)
            f_paths.append(str(out_p))
            f_aligned_list.append(aligned)
            f_all_detected = f_all_detected and ok

        m_paths, m_aligned_list, m_all_detected = [], [], True
        for i, img in enumerate(m_list):
            aligned, ok = self.align_face_image(img)
            out_p = pair_dir / f"mother_{i}.jpg"
            aligned.save(out_p, quality=95)
            m_paths.append(str(out_p))
            m_aligned_list.append(aligned)
            m_all_detected = m_all_detected and ok

        f_path_field = "|".join(f_paths)
        m_path_field = "|".join(m_paths)

        new_row = pd.DataFrame(
            [
                {
                    "family_id": clean_id,
                    "father_name": father_name.strip(),
                    "mother_name": mother_name.strip(),
                    "father_path": f_path_field,
                    "mother_path": m_path_field,
                    "source": "Custom",
                }
            ]
        )

        custom_df = self._load_custom_pairs()
        custom_df = custom_df[custom_df["family_id"] != clean_id]
        custom_df = pd.concat([custom_df, new_row], ignore_index=True)
        custom_df.to_csv(self.custom_csv, index=False)

        if not self.gallery_df.empty:
            keep_mask = ~(
                (self.gallery_df["source"] == "Custom")
                & (self.gallery_df["family_id"] == clean_id)
            )
            if not keep_mask.all():
                keep_idx = torch.tensor(
                    np.where(keep_mask.to_numpy())[0],
                    dtype=torch.long,
                    device=self.device,
                )
                self.gallery_df = self.gallery_df[keep_mask].reset_index(drop=True)
                self.f_emb = self.f_emb[keep_idx]
                self.m_emb = self.m_emb[keep_idx]

        f_vec = self._embed_path_field(f_path_field)
        m_vec = self._embed_path_field(m_path_field)

        self.gallery_df = pd.concat([new_row, self.gallery_df], ignore_index=True)
        self.f_emb = torch.cat([f_vec, self.f_emb], dim=0)
        self.m_emb = torch.cat([m_vec, self.m_emb], dim=0)

        return {
            "family_id": clean_id,
            "father_aligned": f_aligned_list,
            "mother_aligned": m_aligned_list,
            "father_detected": f_all_detected,
            "mother_detected": m_all_detected,
        }

    def remove_parent_couple(self, family_id: str) -> bool:
        custom_df = self._load_custom_pairs()
        if family_id not in set(custom_df["family_id"]):
            return False

        custom_df = custom_df[custom_df["family_id"] != family_id].reset_index(
            drop=True
        )
        custom_df.to_csv(self.custom_csv, index=False)

        pair_dir = self.custom_gallery_dir / family_id
        if pair_dir.is_dir():
            shutil.rmtree(pair_dir, ignore_errors=True)

        if not self.gallery_df.empty:
            keep_mask = ~(
                (self.gallery_df["source"] == "Custom")
                & (self.gallery_df["family_id"] == family_id)
            )
            keep_idx = torch.tensor(
                np.where(keep_mask.to_numpy())[0],
                dtype=torch.long,
                device=self.device,
            )
            self.gallery_df = self.gallery_df[keep_mask].reset_index(drop=True)
            self.f_emb = self.f_emb[keep_idx]
            self.m_emb = self.m_emb[keep_idx]

        return True

    def search(
        self,
        child_input: Union[str, Path, Image.Image],
        top_k: int = 5,
        source_filter: str = "All",
    ) -> List[Dict]:
        if self.gallery_df.empty:
            return []

        if source_filter in ("Custom", "FIW Dataset"):
            mask = (self.gallery_df["source"] == source_filter).to_numpy()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return []
            idx_tensor = torch.tensor(
                valid_indices, dtype=torch.long, device=self.device
            )
            sub_df = self.gallery_df.iloc[valid_indices].reset_index(drop=True)
            f_emb = self.f_emb[idx_tensor]
            m_emb = self.m_emb[idx_tensor]
        else:
            sub_df = self.gallery_df
            f_emb = self.f_emb
            m_emb = self.m_emb

        with torch.no_grad():
            c_emb = self.extract_tta_embedding(child_input)
            c_expanded = c_emb.expand(f_emb.shape[0], -1)
            logits = self.model.forward_from_embeddings(
                f_emb, m_emb, c_expanded
            ).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

        k = min(top_k, len(sub_df))
        ranked_indices = np.argsort(-scores)[:k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = sub_df.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "family_id": row["family_id"],
                    "father_name": row.get("father_name", ""),
                    "mother_name": row.get("mother_name", ""),
                    "father_path": _split_paths(row["father_path"])[0].as_posix(),
                    "mother_path": _split_paths(row["mother_path"])[0].as_posix(),
                    "father_photo_count": len(_split_paths(row["father_path"])),
                    "mother_photo_count": len(_split_paths(row["mother_path"])),
                    "source": row.get("source", "FIW Dataset"),
                }
            )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 1-to-N parent search for a child image"
    )
    parser.add_argument("--child", type=Path, required=False)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--source",
        type=str,
        default="All",
        choices=["All", "Custom", "FIW Dataset"],
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
        target = Path("data/aligned") / Path(*parts[f_idx:])
        print(f"\n[+] Testing with sample child image: {target}")
    else:
        target = args.child

    matches = engine.search(target, top_k=args.top_k, source_filter=args.source)
    for m in matches:
        print(
            f"Rank {m['rank']} | Confidence: {m['score'] * 100:.2f}% | "
            f"Family: {m['family_id']} ({m['source']})"
        )
