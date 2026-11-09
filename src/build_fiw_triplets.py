import argparse
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def is_valid_image(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        with Image.open(p) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_triplets(raw_dir: Path, out_csv: Path, max_combos_per_triad: int = 4):
    # Dynamically find train_relationships.csv and train-faces regardless of nesting
    rel_files = list(raw_dir.rglob("train_relationships.csv"))
    faces_dirs = [d for d in raw_dir.rglob("train-faces") if d.is_dir()]

    if not rel_files:
        raise FileNotFoundError(
            f"Could not find train_relationships.csv anywhere inside {raw_dir}"
        )
    if not faces_dirs:
        raise FileNotFoundError(
            f"Could not find train-faces folder anywhere inside {raw_dir}"
        )

    rel_csv = rel_files[0]
    faces_dir = faces_dirs[0]

    print(f"[+] Found relationships: {rel_csv}")
    print(f"[+] Found faces directory: {faces_dir}")

    df_rel = pd.read_csv(rel_csv).dropna()
    print(f"[+] Loaded {len(df_rel)} relationship rows.")

    # Build adjacency per family
    family_graphs = defaultdict(lambda: defaultdict(set))
    for _, row in df_rel.iterrows():
        p1 = str(row["p1"]).strip()
        p2 = str(row["p2"]).strip()
        if "/" not in p1 or "/" not in p2:
            continue
        f1, m1 = p1.split("/")
        f2, m2 = p2.split("/")
        if f1 == f2:
            family_graphs[f1][m1].add(m2)
            family_graphs[f1][m2].add(m1)

    records = []
    found_triads = set()

    for fam_id, adj in tqdm(family_graphs.items(), desc="Extracting Triads"):
        fam_dir = faces_dir / fam_id
        if not fam_dir.is_dir():
            continue

        members = list(adj.keys())
        for a, b, c in combinations(members, 3):
            # A, B, and C are mutually related (father, mother, child triad)
            if (b in adj[a]) and (c in adj[a]) and (c in adj[b]):
                triad_key = tuple(sorted([a, b, c]))
                if triad_key in found_triads:
                    continue
                found_triads.add(triad_key)

                dir_a, dir_b, dir_c = fam_dir / a, fam_dir / b, fam_dir / c
                if not (dir_a.is_dir() and dir_b.is_dir() and dir_c.is_dir()):
                    continue

                imgs_a = sorted([p for p in dir_a.glob("*.*") if is_valid_image(p)])
                imgs_b = sorted([p for p in dir_b.glob("*.*") if is_valid_image(p)])
                imgs_c = sorted([p for p in dir_c.glob("*.*") if is_valid_image(p)])

                if not (imgs_a and imgs_b and imgs_c):
                    continue

                combos = list(product(imgs_a, imgs_b, imgs_c))[:max_combos_per_triad]
                for fa, mo, ch in combos:
                    records.append(
                        {
                            "family_id": fam_id,
                            "father_path": str(fa.resolve()),
                            "mother_path": str(mo.resolve()),
                            "child_path": str(ch.resolve()),
                            "relation_type": "TRIAD",
                        }
                    )

    if not records:
        print(
            "[!] No triads found. Check that folder names in train-faces match family IDs."
        )
        return

    out_df = pd.DataFrame(records)
    out_df.insert(0, "triplet_id", [f"T{i:05d}" for i in range(len(out_df))])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(
        f"\n[+] SUCCESS: Generated {len(out_df)} triplets across {out_df['family_id'].nunique()} families."
    )
    print(f"[+] Saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw/FIW"))
    parser.add_argument("--out_csv", type=Path, default=Path("data/triplets.csv"))
    args = parser.parse_args()

    build_triplets(args.raw_dir, args.out_csv)
