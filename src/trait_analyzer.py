import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TraitAnalyzer:
    """
    Dynamically identifies facial landmarks (eyes, nose, mouth) using MTCNN
    and measures phenotype inheritance between Child, Father, and Mother.
    """

    def __init__(self, model, mtcnn, device: torch.device):
        self.model = model
        self.mtcnn = mtcnn
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _get_landmarks(
        self, img: Image.Image
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects the primary face bounding box and 5 facial landmarks."""
        try:
            boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
            if landmarks is not None and len(landmarks) > 0:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                best_idx = int(np.argmax(areas))
                return boxes[best_idx], landmarks[best_idx]
        except Exception:
            pass
        return None, None

    def _safe_crop(
        self, img: Image.Image, x1: float, y1: float, x2: float, y2: float
    ) -> Image.Image:
        """Clamps bounding box safely within image dimensions and crops."""
        w, h = img.size
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        return img.crop((x1, y1, x2, y2))

    def _extract_crops(self, img: Image.Image) -> Dict[str, Image.Image]:
        """Extracts dynamic landmark-anchored anatomical regions."""
        box, lm = self._get_landmarks(img)
        w, h = img.size

        if lm is not None:
            # 5 Canonical MTCNN Landmarks:
            # 0: Left Eye, 1: Right Eye, 2: Nose, 3: Left Mouth, 4: Right Mouth
            eye_l, eye_r = lm[0], lm[1]
            nose = lm[2]
            mouth_l, mouth_r = lm[3], lm[4]

            eye_center = (eye_l + eye_r) / 2.0
            eye_dist = max(float(np.linalg.norm(eye_r - eye_l)), 20.0)
            mouth_center = (mouth_l + mouth_r) / 2.0

            # 1. Eyes: Centered on ocular midpoint
            eyes_crop = self._safe_crop(
                img,
                x1=eye_center[0] - eye_dist * 1.05,
                y1=eye_center[1] - eye_dist * 0.40,
                x2=eye_center[0] + eye_dist * 1.05,
                y2=eye_center[1] + eye_dist * 0.40,
            )

            # 2. Nose: Centered on the nasal tip
            nose_crop = self._safe_crop(
                img,
                x1=nose[0] - eye_dist * 0.45,
                y1=nose[1] - eye_dist * 0.50,
                x2=nose[0] + eye_dist * 0.45,
                y2=nose[1] + eye_dist * 0.35,
            )

            # 3. Mouth & Lips: Centered on oral commissures
            mouth_crop = self._safe_crop(
                img,
                x1=mouth_center[0] - eye_dist * 0.70,
                y1=mouth_center[1] - eye_dist * 0.35,
                x2=mouth_center[0] + eye_dist * 0.70,
                y2=mouth_center[1] + eye_dist * 0.45,
            )

            # 4. Hair & Forehead: Positioned above ocular line
            hair_crop = self._safe_crop(
                img,
                x1=eye_center[0] - eye_dist * 1.0,
                y1=eye_center[1] - eye_dist * 1.45,
                x2=eye_center[0] + eye_dist * 1.0,
                y2=eye_center[1] - eye_dist * 0.35,
            )

            # 5. Skin Complexion: Cheek area beside nasal wing
            cheek_x = nose[0] - eye_dist * 0.55
            cheek_y = (eye_center[1] + nose[1]) / 2.0
            r = eye_dist * 0.22
            skin_crop = self._safe_crop(
                img, cheek_x - r, cheek_y - r, cheek_x + r, cheek_y + r
            )

        else:
            # Fallback to proportional face-box cropping if landmarks fail
            if box is not None:
                bx1, by1, bx2, by2 = box
                bw, bh = bx2 - bx1, by2 - by1
            else:
                bx1, by1, bw, bh = 0, 0, w, h

            eyes_crop = self._safe_crop(
                img, bx1 + 0.15 * bw, by1 + 0.20 * bh, bx1 + 0.85 * bw, by1 + 0.42 * bh
            )
            nose_crop = self._safe_crop(
                img, bx1 + 0.28 * bw, by1 + 0.38 * bh, bx1 + 0.72 * bw, by1 + 0.65 * bh
            )
            mouth_crop = self._safe_crop(
                img, bx1 + 0.22 * bw, by1 + 0.62 * bh, bx1 + 0.78 * bw, by1 + 0.88 * bh
            )
            hair_crop = self._safe_crop(
                img, bx1 + 0.15 * bw, by1 + 0.00 * bh, bx1 + 0.85 * bw, by1 + 0.24 * bh
            )
            skin_crop = self._safe_crop(
                img, bx1 + 0.18 * bw, by1 + 0.45 * bh, bx1 + 0.38 * bw, by1 + 0.65 * bh
            )

        return {
            "Eyes": eyes_crop,
            "Nose": nose_crop,
            "Mouth & Lips": mouth_crop,
            "Hair & Forehead": hair_crop,
            "Skin Complexion": skin_crop,
        }

    def _to_tensor(self, crop: Image.Image) -> torch.Tensor:
        resized = crop.resize((160, 160)).convert("RGB")
        return self.transform(resized).unsqueeze(0).to(self.device)

    def _analyze_color_trait(
        self, c_patch: Image.Image, f_patch: Image.Image, m_patch: Image.Image
    ) -> Tuple[float, float]:
        """Calculates perceptual Euclidean distance in YCbCr color space."""
        c_val = np.array(c_patch.convert("YCbCr"), dtype=np.float32).mean(axis=(0, 1))
        f_val = np.array(f_patch.convert("YCbCr"), dtype=np.float32).mean(axis=(0, 1))
        m_val = np.array(m_patch.convert("YCbCr"), dtype=np.float32).mean(axis=(0, 1))

        d_f = np.linalg.norm(c_val - f_val)
        d_m = np.linalg.norm(c_val - m_val)

        inv_f = 1.0 / (d_f + 1e-5)
        inv_m = 1.0 / (d_m + 1e-5)
        f_pct = float(inv_f / (inv_f + inv_m)) * 100.0
        return round(f_pct, 1), round(100.0 - f_pct, 1)

    def analyze_all(
        self, img_f: Image.Image, img_m: Image.Image, img_c: Image.Image
    ) -> Dict[str, dict]:
        f_crops = self._extract_crops(img_f)
        m_crops = self._extract_crops(img_m)
        c_crops = self._extract_crops(img_c)

        results = {}
        structural_traits = ["Eyes", "Nose", "Mouth & Lips", "Hair & Forehead"]

        for trait in structural_traits:
            t_f = self._to_tensor(f_crops[trait])
            t_m = self._to_tensor(m_crops[trait])
            t_c = self._to_tensor(c_crops[trait])

            with torch.no_grad():
                emb_f = self.model.extract_features(t_f)
                emb_m = self.model.extract_features(t_m)
                emb_c = self.model.extract_features(t_c)

                sim_f = F.cosine_similarity(emb_c, emb_f).item()
                sim_m = F.cosine_similarity(emb_c, emb_m).item()

            temp = 0.08
            exp_f = np.exp(sim_f / temp)
            exp_m = np.exp(sim_m / temp)
            f_pct = round(float(exp_f / (exp_f + exp_m)) * 100.0, 1)
            m_pct = round(100.0 - f_pct, 1)

            results[trait] = {
                "father_pct": f_pct,
                "mother_pct": m_pct,
                "dominant": "Father" if f_pct >= 50.0 else "Mother",
                "dominant_pct": max(f_pct, m_pct),
                "crop_f": f_crops[trait],
                "crop_c": c_crops[trait],
                "crop_m": m_crops[trait],
            }

        f_skin_pct, m_skin_pct = self._analyze_color_trait(
            c_crops["Skin Complexion"],
            f_crops["Skin Complexion"],
            m_crops["Skin Complexion"],
        )

        results["Skin Complexion"] = {
            "father_pct": f_skin_pct,
            "mother_pct": m_skin_pct,
            "dominant": "Father" if f_skin_pct >= 50.0 else "Mother",
            "dominant_pct": max(f_skin_pct, m_skin_pct),
            "crop_f": f_crops["Skin Complexion"],
            "crop_c": c_crops["Skin Complexion"],
            "crop_m": m_crops["Skin Complexion"],
        }

        return results
