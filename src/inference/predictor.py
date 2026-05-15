"""
Inference: multi-class chest X-ray + optional severity + optional tabular (age/gender).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, p: Union[str, Path]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _class_folder_order(data_path: Path) -> Optional[List[str]]:
    if not data_path.is_dir():
        return None
    dirs = sorted(d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(dirs) < 2:
        return None
    return [d.name for d in dirs]


def _find_class_index(names: List[str], *keywords: str) -> Optional[int]:
    for i, raw in enumerate(names):
        s = str(raw)
        sl = s.lower()
        if any((k.lower() in sl) or (k in s) for k in keywords):
            return i
    return None


class PneumoniaPredictor:
    """Load config + checkpoint; run single-image inference."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        device: str,
        checkpoint_path: Optional[Path] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        m = config["model"]
        self.model_type = m.get("model_type", "binary")
        self.num_classes = int(m.get("num_classes", 2))
        self.severity_classes = int(m.get("severity_classes", 5))
        self.tabular_dim = int(m.get("tabular_dim", 0))
        dcfg = config.get("data") or {}
        self.tabular_extra_columns: List[str] = list(dcfg.get("tabular_extra_columns") or [])

        data_dir = dcfg.get("data_dir")
        data_path = _resolve_path(PROJECT_ROOT, data_dir) if data_dir else None

        inf = config.get("inference") or {}
        folder_names = _class_folder_order(data_path) if data_path else None
        display = inf.get("class_display_names")
        if display and len(display) == self.num_classes:
            self.class_names: List[str] = list(display)
        elif folder_names and len(folder_names) == self.num_classes:
            self.class_names = folder_names
        else:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

        self._normal_idx: Optional[int] = _find_class_index(self.class_names, "normal", "正常")
        self._virus_idx: Optional[int] = _find_class_index(self.class_names, "virus", "病毒")
        self._bacteria_idx: Optional[int] = _find_class_index(
            self.class_names, "bacteria", "细菌", "bacterium"
        )

        centers = inf.get("severity_bin_centers")
        if centers and len(centers) == self.severity_classes:
            self.severity_bin_centers = np.array(centers, dtype=np.float32)
        else:
            self.severity_bin_centers = np.linspace(10.0, 90.0, self.severity_classes).astype(np.float32)

        image_size = int(dcfg.get("image_size", 224))
        if mean is None or std is None:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        )

        if checkpoint_path and checkpoint_path.is_file():
            ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            try:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            except Exception as e:
                raise RuntimeError(
                    "Checkpoint incompatible with current model config "
                    "(e.g. binary vs multi_task, or tabular_dim mismatch). "
                    "Align configs/config.yaml with how the checkpoint was trained."
                ) from e
            logger.info("Loaded checkpoint %s", checkpoint_path)

    @classmethod
    def from_config_file(
        cls,
        config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_dataset_normalization: bool = True,
    ) -> "PneumoniaPredictor":
        from src.models.medical_models import BinaryClassifier, create_model
        from src.preprocessing.dicom_xray_loader import get_image_statistics

        config_path = _resolve_path(PROJECT_ROOT, config_path)
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = config["model"]

        ck = config.get("paths", {}).get("checkpoint_dir", "checkpoints/")
        ck_path = _resolve_path(PROJECT_ROOT, Path(ck) / "best_model.pth")
        if checkpoint_path:
            ck_path = _resolve_path(PROJECT_ROOT, checkpoint_path)

        mean = std = None
        data_dir = config.get("data", {}).get("data_dir")
        if use_dataset_normalization and data_dir:
            dp = _resolve_path(PROJECT_ROOT, data_dir)
            if dp.is_dir():
                mean, std = get_image_statistics(str(dp))

        model = create_model(
            model_type=m["model_type"],
            backbone=m["name"],
            pretrained=m.get("pretrained", True),
            device=device,
            num_classes=m.get("num_classes"),
            severity_classes=m.get("severity_classes"),
            tabular_dim=int(m.get("tabular_dim", 0)),
        )

        if not ck_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

        ckpt = torch.load(str(ck_path), map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            if any(k.startswith("model.") for k in sd) and not any(k.startswith("backbone.") for k in sd):
                logger.warning(
                    "Classification-only checkpoint (BinaryClassifier). "
                    "Severity will be approximate unless you train multi_task."
                )
                w = sd.get("model.classifier.weight")
                if w is None:
                    raise
                n_cls = int(w.shape[0])
                model = BinaryClassifier(
                    backbone=m["name"],
                    pretrained=False,
                    num_classes=n_cls,
                ).to(device)
                model.load_state_dict(sd, strict=True)
                config = dict(config)
                config["model"] = dict(config["model"])
                config["model"]["model_type"] = "binary"
                config["model"]["num_classes"] = n_cls
                return cls(config, model, device, checkpoint_path=None, mean=mean, std=std)
            raise

        return cls(config, model, device, checkpoint_path=None, mean=mean, std=std)

    def _tensor_from_image_path(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def _tabular_batch(
        self,
        batch_size: int,
        age: Optional[float],
        gender: Optional[str],
        extra: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if self.tabular_dim <= 0:
            return None
        from src.preprocessing.dicom_xray_loader import tabular_vector_from_patient

        vec = tabular_vector_from_patient(
            self.tabular_dim,
            age=age,
            gender=gender,
            extra_columns=self.tabular_extra_columns or None,
            extra_values=extra,
        )
        t = torch.from_numpy(vec).to(self.device).unsqueeze(0).expand(batch_size, -1).float()
        return t

    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path],
        age: Optional[float] = None,
        gender: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        batch = self._tensor_from_image_path(path).unsqueeze(0).to(self.device)
        tab = self._tabular_batch(batch.shape[0], age, gender, extra)
        return self._forward_batch(batch, str(path), tabular=tab)

    def _pneumonia_and_subtype(self, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """probs shape (C,) sums to 1."""
        p = probs.astype(np.float64)
        if self._normal_idx is not None and 0 <= self._normal_idx < len(p):
            p_pneu = float(np.clip(1.0 - p[self._normal_idx], 0.0, 1.0))
        else:
            p_pneu = 1.0

        subtype: Dict[str, float] = {}
        vi, bi = self._virus_idx, self._bacteria_idx
        if vi is not None and bi is not None and vi < len(p) and bi < len(p):
            pv, pb = float(p[vi]), float(p[bi])
            denom = pv + pb
            if denom > 1e-8:
                subtype["p_viral_if_infection"] = pv / denom
                subtype["p_bacterial_if_infection"] = pb / denom
            else:
                subtype["p_viral_if_infection"] = 0.5
                subtype["p_bacterial_if_infection"] = 0.5
            subtype["p_viral_raw"] = pv
            subtype["p_bacterial_raw"] = pb
        return p_pneu, subtype

    def _forward_batch(
        self,
        batch: torch.Tensor,
        path_label: str,
        tabular: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.model_type == "binary":
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            p = probs[0].detach().cpu().numpy()
            p_pneu, subtype = self._pneumonia_and_subtype(p)
            if self._normal_idx is not None and self._normal_idx < len(p):
                rough_sev = float((1.0 - p[self._normal_idx]) * 90.0)
            else:
                rough_sev = float(max(float(x) for x in p) * 90.0)
            out: Dict[str, Any] = {
                "image_path": path_label,
                "model_type": "binary",
                "class_probabilities": {self.class_names[i]: float(p[i]) for i in range(len(p))},
                "predicted_class_index": pred_idx,
                "predicted_class": self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx),
                "pneumonia_probability": round(p_pneu, 4),
                "subtype": {k: round(float(v), 4) for k, v in subtype.items()},
                "severity_estimated_percent": round(rough_sev, 1),
                "severity_note": "Rough severity with classification-only weights; train multi_task for calibrated bins.",
            }
            return out

        class_logits, severity_logits = self.model(batch, tabular)
        class_probs = torch.softmax(class_logits, dim=1)
        sev_probs = torch.softmax(severity_logits, dim=1)

        pred_c = int(torch.argmax(class_probs, dim=1).item())
        pred_s = int(torch.argmax(sev_probs, dim=1).item())

        p = class_probs[0].detach().cpu().numpy()
        p_pneu, subtype = self._pneumonia_and_subtype(p)
        sev_pct = float((sev_probs.cpu().numpy() @ self.severity_bin_centers).item())

        sev_labels = [
            f"bin~{int(self.severity_bin_centers[i])}%"
            for i in range(self.severity_classes)
        ]

        return {
            "image_path": path_label,
            "model_type": "multi_task",
            "class_probabilities": {self.class_names[i]: float(class_probs[0, i].item()) for i in range(class_probs.shape[1])},
            "predicted_class_index": pred_c,
            "predicted_class": self.class_names[pred_c] if pred_c < len(self.class_names) else str(pred_c),
            "pneumonia_probability": round(p_pneu, 4),
            "subtype": {k: round(float(v), 4) for k, v in subtype.items()},
            "severity_bin_probabilities": {sev_labels[i]: float(sev_probs[0, i].item()) for i in range(sev_probs.shape[1])},
            "severity_predicted_bin_index": pred_s,
            "severity_estimated_percent": round(sev_pct, 1),
            "severity_interpretation": self._severity_text(sev_probs.cpu().numpy()[0], pred_s),
        }

    def _severity_text(self, sev_prob: np.ndarray, pred_bin: int) -> str:
        tier = ["很轻", "较轻", "中等", "较重", "很重"]
        base = tier[pred_bin] if pred_bin < len(tier) else f"档位{pred_bin}"
        pct = float(sev_prob @ self.severity_bin_centers)
        return f"综合估计严重程度约 {pct:.1f}%（{base}，该档置信度 {float(sev_prob[pred_bin]):.1%}）"

    @staticmethod
    def result_for_english_ui(result: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(result)
        mt = r.get("model_type")
        if mt == "binary":
            r["severity_note"] = (
                "Rough severity from class probabilities (classification-only checkpoint). "
                "Train multi_task for calibrated severity bins."
            )
        elif mt == "multi_task":
            idx = int(r.get("severity_predicted_bin_index", 0))
            tier = ["minimal", "mild", "moderate", "severe", "critical"]
            name = tier[idx] if idx < len(tier) else f"bin_{idx}"
            pct = r.get("severity_estimated_percent", 0)
            vals = list((r.get("severity_bin_probabilities") or {}).values())
            p_bin = float(vals[idx]) if idx < len(vals) else 0.0
            r["severity_interpretation"] = (
                f"Estimated overall severity ~{pct}% ({name}; bin confidence {p_bin:.1%})."
            )
            old = r.get("severity_bin_probabilities") or {}
            if old:
                vlist = [float(x) for x in old.values()]
                n = len(vlist)
                tiers = [round(10 + i * (80 / max(n - 1, 1))) for i in range(n)] if n > 1 else [50]
                r["severity_bin_probabilities"] = {f"~{tiers[i]}% tier": vlist[i] for i in range(n)}
        return r

    @staticmethod
    def format_report(result: Dict[str, Any], language: str = "zh") -> str:
        if language.lower().startswith("en"):
            return PneumoniaPredictor._format_report_en(result)
        lines = [
            "=" * 52,
            "胸部 X 线辅助分析（研究用途，不能替代医生诊断）",
            "=" * 52,
            f"图像: {result.get('image_path', '')}",
            "",
            "【肺炎（相对正常）总体可能性】",
        ]
        if "pneumonia_probability" in result:
            lines.append(f"  → P(肺炎相关) ≈ {float(result['pneumonia_probability']) * 100:.1f}%")
        lines.extend(["", "【分型概率】"])
        for k, v in result.get("class_probabilities", {}).items():
            lines.append(f"  · {k}: {float(v):.2%}")
        lines.append(f"  → 模型倾向类别: {result.get('predicted_class', '')}")
        sub = result.get("subtype") or {}
        if sub:
            lines.append("")
            lines.append("【病毒/细菌相对倾向（原始与条件概率）】")
            for k, v in sub.items():
                lines.append(f"  · {k}: {float(v):.4f}")
        if result.get("model_type") == "binary" and "severity_estimated_percent" in result:
            lines.append("【严重程度（粗略）】")
            lines.append(f"  → 估计严重度约 {result.get('severity_estimated_percent')}%")
            lines.append(f"  · {result.get('severity_note', '')}")

        if result.get("model_type") == "multi_task":
            lines.append("【严重程度（分档）】")
            for k, v in (result.get("severity_bin_probabilities") or {}).items():
                lines.append(f"  · {k}: {float(v):.2%}")
            lines.append(
                f"  → 综合估计严重度: 约 {result.get('severity_estimated_percent', 0)}% "
                f"（加权于各档代表百分比）"
            )
            lines.append(f"  · {result.get('severity_interpretation', '')}")
        lines.append("=" * 52)
        return "\n".join(lines)

    @staticmethod
    def _format_report_en(result: Dict[str, Any]) -> str:
        lines = [
            "=" * 52,
            "Chest X-ray — assisted analysis (not a medical diagnosis)",
            "=" * 52,
            f"Image: {result.get('image_path', '')}",
            "",
            "Pneumonia-related probability (vs normal, if a normal class exists)",
        ]
        if "pneumonia_probability" in result:
            lines.append(f"  → {float(result['pneumonia_probability']) * 100:.1f}%")
        lines.extend(["", "Class probabilities"])
        for k, v in result.get("class_probabilities", {}).items():
            lines.append(f"  · {k}: {float(v):.2%}")
        lines.append(f"  → Predicted class: {result.get('predicted_class', '')}")
        sub = result.get("subtype") or {}
        if sub:
            lines.append("")
            lines.append("Subtype")
            for k, v in sub.items():
                lines.append(f"  · {k}: {float(v):.4f}")
        lines.append("")
        if result.get("model_type") == "binary" and "severity_estimated_percent" in result:
            lines.append("Severity (rough)")
            lines.append(f"  → ~{result.get('severity_estimated_percent')}%")
            lines.append(f"  · {result.get('severity_note', '')}")
        if result.get("model_type") == "multi_task":
            lines.append("Severity (bins)")
            for k, v in (result.get("severity_bin_probabilities") or {}).items():
                lines.append(f"  · {k}: {float(v):.2%}")
            lines.append(f"  → Blended severity ~{result.get('severity_estimated_percent', 0)}%")
            lines.append(f"  · {result.get('severity_interpretation', '')}")
        lines.append("=" * 52)
        return "\n".join(lines)

    @staticmethod
    def to_json(result: Dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False, indent=2)
