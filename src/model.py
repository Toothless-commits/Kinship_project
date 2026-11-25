import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class TriSubjectKinshipNet(nn.Module):
    def __init__(self, pretrained: str = "vggface2", freeze_backbone: bool = True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Shared low-dimensional bottleneck prevents overfitting on small triad datasets
        self.proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
        )

        # 5 symmetric 128-d interaction tensors (640) + 6 direct 512-d cosine scalars = 646
        in_features = 128 * 5 + 6

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def unfreeze_top_backbone(self):
        """Unfreezes only the final InceptionResnetV1 projection layers."""
        top_prefixes = ("block8", "last_linear", "last_bn")
        for name, param in self.backbone.named_parameters():
            if name.startswith(top_prefixes):
                param.requires_grad = True

    def extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(img_tensor)
        return F.normalize(feat, p=2, dim=1)

    def fuse_embeddings(
        self,
        feat_f: torch.Tensor,
        feat_m: torch.Tensor,
        feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines parameter-free 512-d cosine geometry with compact symmetric
        interactions in a shared 128-d latent space.
        """
        # 1. Exact cosine similarities in pretrained 512-d face space
        mid_512 = F.normalize(feat_f + feat_m, p=2, dim=1)
        cos_fc = (feat_f * feat_c).sum(dim=1, keepdim=True)
        cos_mc = (feat_m * feat_c).sum(dim=1, keepdim=True)
        cos_mid = (mid_512 * feat_c).sum(dim=1, keepdim=True)
        cos_max = torch.maximum(cos_fc, cos_mc)
        cos_min = torch.minimum(cos_fc, cos_mc)
        cos_fm = (feat_f * feat_m).sum(dim=1, keepdim=True)

        # 2. Compact symmetric fusion in 128-d projected space
        pf = F.normalize(self.proj(feat_f), p=2, dim=1)
        pm = F.normalize(self.proj(feat_m), p=2, dim=1)
        pc = F.normalize(self.proj(feat_c), p=2, dim=1)
        mid_128 = F.normalize(pf + pm, p=2, dim=1)

        diff_fc = torch.abs(pf - pc)
        diff_mc = torch.abs(pm - pc)
        diff_min = torch.minimum(diff_fc, diff_mc)
        diff_max = torch.maximum(diff_fc, diff_mc)

        prod_fc = pf * pc
        prod_mc = pm * pc
        prod_max = torch.maximum(prod_fc, prod_mc)
        prod_avg = 0.5 * (prod_fc + prod_mc)

        diff_mid = torch.abs(mid_128 - pc)

        return torch.cat(
            [
                cos_max,
                cos_min,
                cos_mid,
                cos_fm,
                (cos_fc + cos_mc) * 0.5,
                torch.abs(cos_fc - cos_mc),
                diff_min,
                diff_max,
                prod_max,
                prod_avg,
                diff_mid,
            ],
            dim=1,
        )

    def forward_from_embeddings(
        self,
        feat_f: torch.Tensor,
        feat_m: torch.Tensor,
        feat_c: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse_embeddings(feat_f, feat_m, feat_c)
        return self.classifier(fused)

    def forward(
        self,
        img_father: torch.Tensor,
        img_mother: torch.Tensor,
        img_child: torch.Tensor,
    ) -> torch.Tensor:
        feat_f = self.extract_features(img_father)
        feat_m = self.extract_features(img_mother)
        feat_c = self.extract_features(img_child)
        return self.forward_from_embeddings(feat_f, feat_m, feat_c)
