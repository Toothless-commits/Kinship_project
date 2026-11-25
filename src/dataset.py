import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def map_to_aligned_path(raw_path_str: str, aligned_root: Path) -> Path:
    p = Path(raw_path_str)
    parts = p.parts
    try:
        f_idx = next(
            i
            for i, part in enumerate(parts)
            if part.startswith("F") and len(part) >= 4 and part[1:].isdigit()
        )
        return aligned_root / Path(*parts[f_idx:])
    except StopIteration:
        return p


def split_families(
    csv_path: Path,
    aligned_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    df = pd.read_csv(csv_path)

    df["father_path"] = df["father_path"].apply(
        lambda p: str(map_to_aligned_path(p, aligned_root))
    )
    df["mother_path"] = df["mother_path"].apply(
        lambda p: str(map_to_aligned_path(p, aligned_root))
    )
    df["child_path"] = df["child_path"].apply(
        lambda p: str(map_to_aligned_path(p, aligned_root))
    )

    valid_mask = (
        df["father_path"].apply(lambda p: Path(p).is_file())
        & df["mother_path"].apply(lambda p: Path(p).is_file())
        & df["child_path"].apply(lambda p: Path(p).is_file())
    )
    df = df[valid_mask].reset_index(drop=True)

    unique_families = sorted(df["family_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_families)

    n_total = len(unique_families)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_fams = set(unique_families[:n_train])
    val_fams = set(unique_families[n_train : n_train + n_val])
    test_fams = set(unique_families[n_train + n_val :])

    train_df = df[df["family_id"].isin(train_fams)].reset_index(drop=True)
    val_df = df[df["family_id"].isin(val_fams)].reset_index(drop=True)
    test_df = df[df["family_id"].isin(test_fams)].reset_index(drop=True)

    return train_df, val_df, test_df


class KinshipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        self.fam_fathers = defaultdict(list)
        self.fam_mothers = defaultdict(list)
        for _, row in self.df.iterrows():
            fid = row["family_id"]
            self.fam_fathers[fid].append(row["father_path"])
            self.fam_mothers[fid].append(row["mother_path"])

        self.unique_fams = sorted(self.fam_fathers.keys())

        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=8, translate=(0.04, 0.04), scale=(0.94, 1.06)
                    ),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.df) * 2

    def _load_img(self, path_str: str) -> torch.Tensor:
        with Image.open(path_str).convert("RGB") as img:
            return self.transform(img)

    def _sample_impostor_family(self, target_fam: str, rand_gen) -> str:
        if len(self.unique_fams) <= 1:
            return target_fam
        cand = rand_gen.choice(self.unique_fams)
        while cand == target_fam:
            cand = rand_gen.choice(self.unique_fams)
        return cand

    def __getitem__(self, idx: int):
        is_positive = idx % 2 == 0
        triplet_idx = idx // 2
        row = self.df.iloc[triplet_idx]

        f_path = row["father_path"]
        m_path = row["mother_path"]
        c_path = row["child_path"]
        target_fam = row["family_id"]

        if not is_positive:
            rand_gen = random if self.is_train else random.Random(idx)
            # Balanced negatives: 50% wrong couple, 25% wrong father, 25% wrong mother
            r = rand_gen.random()
            if r < 0.25:
                neg_fam = self._sample_impostor_family(target_fam, rand_gen)
                f_path = rand_gen.choice(self.fam_fathers[neg_fam])
            elif r < 0.50:
                neg_fam = self._sample_impostor_family(target_fam, rand_gen)
                m_path = rand_gen.choice(self.fam_mothers[neg_fam])
            else:
                neg_fam = self._sample_impostor_family(target_fam, rand_gen)
                f_path = rand_gen.choice(self.fam_fathers[neg_fam])
                m_path = rand_gen.choice(self.fam_mothers[neg_fam])
            label = 0.0
        else:
            label = 1.0

        f_tensor = self._load_img(f_path)
        m_tensor = self._load_img(m_path)
        c_tensor = self._load_img(c_path)

        return f_tensor, m_tensor, c_tensor, torch.tensor([label], dtype=torch.float32)
