"""
Medical models for pneumonia detection.
Supports DenseNet multi-task (class + severity) with optional tabular (age/gender/extras).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(
    model_type: str,
    backbone: str,
    pretrained: bool = True,
    device: str = "cuda",
    num_classes: Optional[int] = None,
    severity_classes: Optional[int] = None,
    tabular_dim: int = 0,
) -> nn.Module:
    """Create pneumonia model: multi_task (DenseNet) or binary (ResNet/DenseNet)."""
    if model_type == "multi_task":
        model = DenseNetMultiTask(
            num_classes=num_classes or 3,
            severity_classes=severity_classes or 5,
            pretrained=pretrained,
            tabular_dim=int(tabular_dim or 0),
        )
    elif model_type == "binary":
        model = BinaryClassifier(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes or 2,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


class DenseNetMultiTask(nn.Module):
    """Multi-task DenseNet: multi-class chest X-ray + severity bins; optional tabular fusion."""

    def __init__(
        self,
        num_classes: int = 3,
        severity_classes: int = 5,
        pretrained: bool = True,
        tabular_dim: int = 0,
    ):
        super().__init__()
        self.tabular_dim = int(tabular_dim)
        self.backbone = models.densenet121(pretrained=pretrained)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        fused_in = num_features
        if self.tabular_dim > 0:
            self.tabular_proj = nn.Sequential(
                nn.Linear(self.tabular_dim, 64),
                nn.ReLU(inplace=True),
            )
            fused_in = num_features + 64
        else:
            self.tabular_proj = None

        self.binary_head = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, severity_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if self.tabular_proj is not None:
            if tabular is None:
                tabular = torch.zeros(
                    x.shape[0],
                    self.tabular_dim,
                    device=x.device,
                    dtype=features.dtype,
                )
            features = torch.cat([features, self.tabular_proj(tabular)], dim=1)
        class_logits = self.binary_head(features)
        severity_logits = self.severity_head(features)
        return class_logits, severity_logits


class BinaryClassifier(nn.Module):
    """Single-head classifier (folder classes or binary)."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        if backbone == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif backbone == "densenet121":
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
