import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import KinshipDataset, split_families
from model import TriSubjectKinshipNet


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def extract_tta_batch(model, tensor_batch: torch.Tensor) -> torch.Tensor:
    emb_orig = model.extract_features(tensor_batch)
    emb_flip = model.extract_features(torch.flip(tensor_batch, dims=[3]))
    return F.normalize(emb_orig + emb_flip, p=2, dim=1)


def evaluate_binary_verification(model, test_df, batch_size, device):
    test_ds = KinshipDataset(test_df, is_train=False, seed=99)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    all_probs, all_labels = [], []

    with torch.no_grad():
        for f, m, c, y in tqdm(test_loader, desc="1/2 Binary Verification (TTA)"):
            f, m, c = f.to(device), m.to(device), c.to(device)
            ef = extract_tta_batch(model, f)
            em = extract_tta_batch(model, m)
            ec = extract_tta_batch(model, c)
            logits = model.forward_from_embeddings(ef, em, ec)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.extend(probs)
            all_labels.extend(y.numpy().reshape(-1))

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, preds)
    mean_kin_conf = all_probs[all_labels == 1].mean() * 100
    mean_nonkin_conf = all_probs[all_labels == 0].mean() * 100

    print("\n" + "=" * 55)
    print("       PART 1: BINARY VERIFICATION METRICS (TTA)")
    print("=" * 55)
    print(f"Test Accuracy       : {acc * 100:.2f}%")
    print(f"Test ROC-AUC        : {auc:.4f}")
    print(f"Mean Kin Score      : {mean_kin_conf:.2f}%")
    print(f"Mean Non-Kin Score  : {mean_nonkin_conf:.2f}%")
    print("\nConfusion Matrix:")
    print(f"                 Pred Non-Kin    Pred Kin")
    print(f"Actual Non-Kin : {cm[0, 0]:<15} {cm[0, 1]}")
    print(f"Actual Kin     : {cm[1, 0]:<15} {cm[1, 1]}")
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, preds, target_names=["Non-Kin (0)", "Kin (1)"]
        )
    )


def evaluate_rank_k_retrieval(model, test_df, device):
    print("\n" + "=" * 55)
    print("       PART 2: RANK-K PARENT RETRIEVAL METRICS (TTA)")
    print("=" * 55)

    transform = get_eval_transform()

    def load_tensor(path_str):
        with Image.open(path_str).convert("RGB") as img:
            return transform(img)

    parent_pairs = (
        test_df[["family_id", "father_path", "mother_path"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    num_candidates = len(parent_pairs)
    print(f"[+] Total candidate parent pairs in gallery: {num_candidates}")

    f_stack = torch.stack([load_tensor(p) for p in parent_pairs["father_path"]]).to(
        device
    )
    m_stack = torch.stack([load_tensor(p) for p in parent_pairs["mother_path"]]).to(
        device
    )
    parent_families = parent_pairs["family_id"].tolist()

    with torch.no_grad():
        f_emb = torch.cat(
            [extract_tta_batch(model, b) for b in torch.split(f_stack, 32)], dim=0
        )
        m_emb = torch.cat(
            [extract_tta_batch(model, b) for b in torch.split(m_stack, 32)], dim=0
        )

    child_queries = (
        test_df[["family_id", "child_path"]].drop_duplicates().reset_index(drop=True)
    )
    print(f"[+] Total child test queries: {len(child_queries)}")

    hits_top1, hits_top5, mrr_total = 0, 0, 0.0

    with torch.no_grad():
        for _, row in tqdm(
            child_queries.iterrows(),
            total=len(child_queries),
            desc="2/2 Rank-K Queries",
        ):
            true_fam = row["family_id"]
            c_tensor = load_tensor(row["child_path"]).unsqueeze(0).to(device)
            c_emb = extract_tta_batch(model, c_tensor).expand(num_candidates, -1)

            logits = model.forward_from_embeddings(f_emb, m_emb, c_emb)
            scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            ranked_indices = np.argsort(-scores)
            ranked_families = [parent_families[idx] for idx in ranked_indices]

            try:
                first_hit_rank = ranked_families.index(true_fam) + 1
            except ValueError:
                first_hit_rank = None

            if first_hit_rank is not None:
                if first_hit_rank <= 1:
                    hits_top1 += 1
                if first_hit_rank <= 5:
                    hits_top5 += 1
                mrr_total += 1.0 / first_hit_rank

    n_queries = len(child_queries)
    print("\n" + "-" * 55)
    print(
        f"Top-1 Retrieval Accuracy : {(hits_top1 / n_queries) * 100:.2f}% ({hits_top1}/{n_queries})"
    )
    print(
        f"Top-5 Retrieval Accuracy : {(hits_top5 / n_queries) * 100:.2f}% ({hits_top5}/{n_queries})"
    )
    print(f"Mean Reciprocal Rank (MRR): {mrr_total / n_queries:.4f}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/best_model.pt"))
    parser.add_argument("--csv", type=Path, default=Path("data/triplets.csv"))
    parser.add_argument("--aligned", type=Path, default=Path("data/aligned"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df = split_families(args.csv, args.aligned, seed=42)

    model = TriSubjectKinshipNet(freeze_backbone=False)
    model.load_state_dict(torch.load(args.weights, map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()

    evaluate_binary_verification(model, test_df, args.batch_size, dev)
    evaluate_rank_k_retrieval(model, test_df, dev)


if __name__ == "__main__":
    main()
