import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class TriSubjectKinshipNet(nn.Module):
    def __init__(self, pretrained: str = "vggface2", freeze_backbone: bool = True):
        super().__init__()
        # 1. Shared Feature Extractor
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Input dimension after fusion:
        # |f_F - f_C| (512) + |f_M - f_C| (512) +
        # (f_F * f_C) (512) + (f_M * f_C) (512) +
        # |mid_parent - f_C| (512) = 2560 dims
        in_features = 512 * 5

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(img_tensor)
        return torch.nn.functional.normalize(feat, p=2, dim=1)

    def forward(
        self,
        img_father: torch.Tensor,
        img_mother: torch.Tensor,
        img_child: torch.Tensor,
    ) -> torch.Tensor:
        # Extract normalized 512-d embeddings
        feat_f = self.extract_features(img_father)
        feat_m = self.extract_features(img_mother)
        feat_c = self.extract_features(img_child)

        # Pairwise differences
        diff_fc = torch.abs(feat_f - feat_c)
        diff_mc = torch.abs(feat_m - feat_c)

        # Pairwise products
        prod_fc = feat_f * feat_c
        prod_mc = feat_m * feat_c

        # Mid-parent vector to child
        mid_parent = (feat_f + feat_m) / 2.0
        diff_midc = torch.abs(mid_parent - feat_c)

        # Fused genetic representation
        fused = torch.cat([diff_fc, diff_mc, prod_fc, prod_mc, diff_midc], dim=1)

        # Return logits (BCEWithLogitsLoss handles numerical stability)
        return self.classifier(fused)
