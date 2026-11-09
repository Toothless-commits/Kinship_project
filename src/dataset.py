import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def map_to_aligned_path(raw_path_str: str, aligned_root: Path) -> Path:
    p = Path(raw_path_str)
    parts = p.parts
    f_idx = next(
        i
        for i, part in enumerate(parts)
        if part.startswith("F") and len(part) >= 4 and part[1:].isdigit()
    )
    rel_path = Path(*parts[f_idx:])
    return aligned_root / rel_path


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

        self.all_fathers = self.df["father_path"].tolist()
        self.all_mothers = self.df["mother_path"].tolist()
        self.family_ids = self.df["family_id"].tolist()
        self.rng = random.Random(seed)

        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.15, saturation=0.1
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
            neg_type = rand_gen.choice(["wrong_father", "wrong_mother", "wrong_both"])

            cand_f = [
                f
                for f, fam in zip(self.all_fathers, self.family_ids)
                if fam != target_fam
            ]
            cand_m = [
                m
                for m, fam in zip(self.all_mothers, self.family_ids)
                if fam != target_fam
            ]

            if neg_type == "wrong_father":
                f_path = (
                    rand_gen.choice(cand_f)
                    if cand_f
                    else rand_gen.choice(self.all_fathers)
                )
            elif neg_type == "wrong_mother":
                m_path = (
                    rand_gen.choice(cand_m)
                    if cand_m
                    else rand_gen.choice(self.all_mothers)
                )
            else:
                f_path = (
                    rand_gen.choice(cand_f)
                    if cand_f
                    else rand_gen.choice(self.all_fathers)
                )
                m_path = (
                    rand_gen.choice(cand_m)
                    if cand_m
                    else rand_gen.choice(self.all_mothers)
                )

            label = 0.0
        else:
            label = 1.0

        f_tensor = self._load_img(f_path)
        m_tensor = self._load_img(m_path)
        c_tensor = self._load_img(c_path)

        return f_tensor, m_tensor, c_tensor, torch.tensor([label], dtype=torch.float32)
