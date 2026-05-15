"""
Medical Image Loading and Preprocessing

Handles:
- DICOM, JPG, PNG X-ray image formats
- Folder-based or CSV-based labels
- Optional per-image tabular features (age, gender, extra columns from metadata CSV)
- Optional severity labels (CSV, synthetic, or auto by class index)
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".dcm", ".bmp", ".tif", ".tiff", ".webp")


def load_dicom(dicom_path: str) -> np.ndarray:
    """Load DICOM file and extract pixel array (BGR uint8)."""
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array
    pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(pixel_array.shape) == 2:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    return pixel_array


def load_image(image_path: str) -> np.ndarray:
    """Load image from file (BGR)."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def get_image_statistics(image_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std of dataset for normalization."""
    logger.info("Calculating image statistics for %s...", image_dir)
    images: List[np.ndarray] = []
    image_dir = str(image_dir)
    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith(".dcm"):
                    img = load_dicom(file_path)
                else:
                    img = load_image(file_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
            except Exception:
                continue
    if not images:
        logger.warning("No images found. Using default ImageNet statistics.")
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    arr = np.array(images) / 255.0
    mean = arr.mean(axis=(0, 1, 2))
    std = arr.std(axis=(0, 1, 2))
    logger.info("Mean: %s, Std: %s", mean, std)
    return mean, std


def _parse_gender_one_hot(raw: Any) -> Tuple[float, float]:
    """Return (male, female) one-hot; (0,0) if unknown."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0.0, 0.0
    s = str(raw).strip().lower()
    if s in ("m", "male", "1", "男"):
        return 1.0, 0.0
    if s in ("f", "female", "0", "女"):
        return 0.0, 1.0
    return 0.0, 0.0


def _load_metadata_rows(
    metadata_csv: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load CSV keyed by basename -> dict with age, gender, severity, extras."""
    import pandas as pd

    path = Path(metadata_csv)
    if project_root and not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        logger.warning("metadata_csv not found: %s", path)
        return {}
    df = pd.read_csv(path)
    name_col = None
    for c in ("image_name", "filename", "file", "path"):
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError(
            f"metadata_csv must contain one of columns: image_name, filename, file, path. Got: {list(df.columns)}"
        )
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = Path(str(row[name_col])).name
        rec: Dict[str, Any] = {}
        if "age" in df.columns:
            try:
                rec["age"] = float(row["age"])
            except (TypeError, ValueError):
                rec["age"] = None
        if "gender" in df.columns:
            rec["gender"] = row["gender"]
        if "severity" in df.columns:
            try:
                rec["severity"] = int(row["severity"])
            except (TypeError, ValueError):
                rec["severity"] = None
        for col in df.columns:
            if col == name_col or col in ("age", "gender", "severity"):
                continue
            rec[col] = row[col]
        out[key] = rec
    return out


def _build_tabular_vector(
    rec: Optional[Dict[str, Any]],
    tabular_dim: int,
    extra_columns: Optional[List[str]],
) -> np.ndarray:
    """Fixed layout: [age/100, male, female] then optional extra columns in order."""
    vec = np.zeros((tabular_dim,), dtype=np.float32)
    if tabular_dim <= 0:
        return vec
    age = 0.0
    gm, gf = 0.0, 0.0
    if rec:
        if rec.get("age") is not None:
            try:
                age = float(rec["age"]) / 100.0
            except (TypeError, ValueError):
                age = 0.0
        gm, gf = _parse_gender_one_hot(rec.get("gender"))
    idx = 0
    if tabular_dim >= 1:
        vec[idx] = np.clip(age, 0.0, 1.5)
        idx += 1
    if tabular_dim >= 2:
        vec[idx] = gm
        idx += 1
    if tabular_dim >= 3:
        vec[idx] = gf
        idx += 1
    if extra_columns and rec:
        for col in extra_columns:
            if idx >= tabular_dim:
                break
            if col not in rec:
                idx += 1
                continue
            v = rec[col]
            try:
                vec[idx] = float(v)
            except (TypeError, ValueError):
                vec[idx] = 0.0
            idx += 1
    return vec


def tabular_vector_from_patient(
    tabular_dim: int,
    age: Optional[float] = None,
    gender: Optional[str] = None,
    extra_columns: Optional[List[str]] = None,
    extra_values: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build the same tabular vector layout as training for inference."""
    rec: Dict[str, Any] = {}
    if age is not None:
        rec["age"] = float(age)
    if gender is not None:
        rec["gender"] = gender
    if extra_values:
        rec.update(extra_values)
    return _build_tabular_vector(rec if rec else None, tabular_dim, extra_columns)


class XrayDataset(Dataset):
    """Chest X-ray images with optional severity and tabular features."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        severity_labels: Optional[List[int]] = None,
        tabular_vectors: Optional[np.ndarray] = None,
        image_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.severity_labels = list(severity_labels) if severity_labels is not None else None
        self.tabular_vectors = tabular_vectors
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406])
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225])
        self._set_transforms()

    def _set_transforms(self) -> None:
        if self.augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
                    if self.normalize
                    else transforms.Lambda(lambda x: x),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
                    if self.normalize
                    else transforms.Lambda(lambda x: x),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        if image_path.lower().endswith(".dcm"):
            image = load_dicom(image_path)
        else:
            image = load_image(image_path)
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        sample: Dict[str, Any] = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
        }
        if self.severity_labels is not None:
            sample["severity"] = torch.tensor(self.severity_labels[idx], dtype=torch.long)
        if self.tabular_vectors is not None:
            sample["tabular"] = torch.from_numpy(self.tabular_vectors[idx].copy())
        return sample


def _severity_from_labels_array(
    labels: np.ndarray,
    severity_strategy: str,
    synthetic_severity_by_class: Optional[List[int]],
    severity_classes: int,
) -> List[int]:
    """Build per-image severity list aligned with labels (synthetic / auto)."""
    n_cls = int(labels.max()) + 1 if len(labels) else 0
    strat = (severity_strategy or "auto").lower()
    if strat in ("none", "from_csv"):
        raise ValueError("Preset split loader needs severity_strategy auto/synthetic or metadata severity.")
    syn = synthetic_severity_by_class
    if not syn:
        if n_cls <= 1:
            syn = [0]
        else:
            syn = [
                min(int(round(i * (severity_classes - 1) / max(n_cls - 1, 1))), severity_classes - 1)
                for i in range(n_cls)
            ]
        logger.info("Auto synthetic_severity_by_class (per class id): %s", syn)
    if len(syn) < n_cls:
        raise ValueError(f"synthetic_severity_by_class length {len(syn)} < num classes {n_cls}")
    sev_arr = np.array([syn[int(lab)] for lab in labels], dtype=np.int64)
    return sev_arr.tolist()


def _collect_split_class_images(split_root: Path, class_names: List[str], class_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labs: List[int] = []
    for name in class_names:
        idx = class_to_idx[name]
        cdir = split_root / name
        if not cdir.is_dir():
            continue
        for r, _, files in os.walk(str(cdir)):
            for file in files:
                if not file.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                paths.append(os.path.join(r, file))
                labs.append(idx)
    return paths, labs


def _try_collect_preset_train_val_test(
    data_root: Path,
) -> Optional[Tuple[List[str], Dict[str, int], List[str], List[int], List[str], List[int], List[str], List[int]]]:
    """
    If data_root has train/ and val/ each with the same class subfolders, return layout.
    test/ is optional. Returns (class_names_sorted, class_to_idx, tr_p, tr_l, va_p, va_l, te_p, te_l).
    """
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        return None
    class_dirs = sorted(
        d.name for d in train_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(class_dirs) < 2:
        return None
    for c in class_dirs:
        if not (val_root / c).is_dir():
            logger.warning(
                "Dataset has train/ and val/ but val/ is missing class folder %r; using flat layout instead.",
                c,
            )
            return None
    class_to_idx = {n: i for i, n in enumerate(class_dirs)}
    tr_p, tr_l = _collect_split_class_images(train_root, class_dirs, class_to_idx)
    va_p, va_l = _collect_split_class_images(val_root, class_dirs, class_to_idx)
    test_root = data_root / "test"
    if test_root.is_dir():
        te_p, te_l = _collect_split_class_images(test_root, class_dirs, class_to_idx)
    else:
        te_p, te_l = [], []
    if not tr_p or not va_p:
        logger.warning("Preset split layout found but train or val has no images; falling back.")
        return None
    return (class_dirs, class_to_idx, tr_p, tr_l, va_p, va_l, te_p, te_l)


def _build_tabular_matrix_for_paths(
    paths: List[str],
    tabular_dim: int,
    tabular_extra_columns: Optional[List[str]],
    meta_by_basename: Dict[str, Dict[str, Any]],
) -> Optional[np.ndarray]:
    if not tabular_dim or tabular_dim <= 0:
        return None
    rows = [_build_tabular_vector(meta_by_basename.get(Path(p).name), tabular_dim, tabular_extra_columns) for p in paths]
    return np.stack(rows, axis=0)


def _dataloaders_from_explicit_splits(
    data_dir: str,
    tr_paths: List[str],
    tr_labels: List[int],
    va_paths: List[str],
    va_labels: List[int],
    te_paths: List[str],
    te_labels: List[int],
    batch_size: int,
    image_size: int,
    num_workers: int,
    augment_train: bool,
    severity_strategy: str,
    synthetic_severity_by_class: Optional[List[int]],
    severity_classes: int,
    metadata_csv: Optional[str],
    tabular_dim: int,
    tabular_extra_columns: Optional[List[str]],
    project_root: Optional[Path],
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build loaders when train/val paths are already fixed (preset split folders)."""
    meta_by_basename: Dict[str, Dict[str, Any]] = {}
    if metadata_csv:
        meta_by_basename = _load_metadata_rows(metadata_csv, project_root=project_root)

    def _filter(paths: List[str], labels: List[int]) -> Tuple[List[str], np.ndarray, List[int]]:
        vi = [i for i, p in enumerate(paths) if os.path.exists(p)]
        pp = [paths[i] for i in vi]
        lab = np.array([labels[i] for i in vi], dtype=np.int64)
        sev = _severity_from_labels_array(lab, severity_strategy, synthetic_severity_by_class, severity_classes)
        return pp, lab, sev

    tr_paths, tr_lab, tr_sev = _filter(tr_paths, tr_labels)
    va_paths, va_lab, va_sev = _filter(va_paths, va_labels)
    te_paths_f, te_lab, te_sev = _filter(te_paths, te_labels) if te_paths else ([], np.array([], dtype=np.int64), [])

    logger.info(
        "Preset splits: train=%s val=%s test=%s images",
        len(tr_paths),
        len(va_paths),
        len(te_paths_f),
    )

    mean, std = get_image_statistics(data_dir)

    tr_tab = _build_tabular_matrix_for_paths(tr_paths, tabular_dim, tabular_extra_columns, meta_by_basename)
    va_tab = _build_tabular_matrix_for_paths(va_paths, tabular_dim, tabular_extra_columns, meta_by_basename)
    te_tab = (
        _build_tabular_matrix_for_paths(te_paths_f, tabular_dim, tabular_extra_columns, meta_by_basename)
        if te_paths_f
        else None
    )

    train_ds = XrayDataset(
        tr_paths,
        tr_lab.tolist(),
        tr_sev,
        tabular_vectors=tr_tab,
        image_size=image_size,
        augment=augment_train,
        mean=mean,
        std=std,
    )
    val_ds = XrayDataset(
        va_paths,
        va_lab.tolist(),
        va_sev,
        tabular_vectors=va_tab,
        image_size=image_size,
        augment=False,
        mean=mean,
        std=std,
    )
    test_ds = None
    if te_paths_f:
        test_ds = XrayDataset(
            te_paths_f,
            te_lab.tolist(),
            te_sev,
            tabular_vectors=te_tab,
            image_size=image_size,
            augment=False,
            mean=mean,
            std=std,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader


def create_data_loaders(
    data_dir: str,
    csv_file: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    augment_train: bool = True,
    seed: int = 42,
    severity_strategy: str = "auto",
    synthetic_severity_by_class: Optional[List[int]] = None,
    severity_classes: int = 5,
    metadata_csv: Optional[str] = None,
    tabular_dim: int = 0,
    tabular_extra_columns: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train/val/test loaders. Uses disjoint XrayDataset copies so val/test are not augmented.

    severity_strategy:
      - none: severity only from CSV column or metadata_csv; raises if still missing for multi-task.
      - synthetic: use synthetic_severity_by_class per class index (auto-filled if omitted).
      - auto: alias of synthetic with auto-generated mapping.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = str(data_dir)
    image_paths: List[str] = []
    labels_list: List[int] = []
    severity_list: Optional[List[int]] = None
    meta_by_basename: Dict[str, Dict[str, Any]] = {}
    if metadata_csv:
        meta_by_basename = _load_metadata_rows(metadata_csv, project_root=project_root)

    if csv_file and os.path.exists(str(csv_file)):
        import pandas as pd

        logger.info("Loading data from CSV %s", csv_file)
        df = pd.read_csv(csv_file)
        image_paths = [os.path.join(data_dir, str(img)) for img in df["image_name"]]
        labels_list = [int(x) for x in df["label"].values.tolist()]
        if "severity" in df.columns:
            severity_list = [int(x) for x in df["severity"].values.tolist()]
    else:
        root = Path(data_dir)
        preset = _try_collect_preset_train_val_test(root)
        if preset is not None:
            class_names, _idx, tr_p, tr_l, va_p, va_l, te_p, te_l = preset
            logger.info("Using preset train/val[/test] folders; classes=%s", class_names)
            return _dataloaders_from_explicit_splits(
                data_dir,
                tr_p,
                tr_l,
                va_p,
                va_l,
                te_p,
                te_l,
                batch_size,
                image_size,
                num_workers,
                augment_train,
                severity_strategy,
                synthetic_severity_by_class,
                severity_classes,
                metadata_csv,
                tabular_dim,
                tabular_extra_columns,
                project_root,
            )
        class_dirs = sorted(
            p for p in Path(data_dir).iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if len(class_dirs) < 2:
            raise ValueError(
                f"Folder-based dataset expects at least 2 class subfolders under: {data_dir}\n"
                f"Found: {[p.name for p in class_dirs]}"
            )
        class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}
        logger.info("Detected classes: %s", class_to_idx)
        for class_dir in class_dirs:
            for root, _, files in os.walk(str(class_dir)):
                for file in files:
                    if not file.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    image_paths.append(os.path.join(root, file))
                    labels_list.append(class_to_idx[class_dir.name])

    labels = np.array(labels_list, dtype=np.int64)
    n_cls = int(labels.max()) + 1 if len(labels) else 0

    if severity_list is None:
        sev_from_meta: Optional[List[int]] = None
        if meta_by_basename:
            tmp: List[int] = []
            ok = True
            for p in image_paths:
                rec = meta_by_basename.get(Path(p).name)
                if not rec or rec.get("severity") is None:
                    ok = False
                    break
                tmp.append(int(rec["severity"]))
            if ok and len(tmp) == len(image_paths):
                sev_from_meta = tmp
        if sev_from_meta is not None:
            severity_list = sev_from_meta
        else:
            strat = (severity_strategy or "auto").lower()
            if strat in ("none", "from_csv"):
                raise ValueError(
                    "No severity labels found (CSV/metadata). Set data.severity_strategy to "
                    "'auto' or 'synthetic', or provide severity in CSV / metadata_csv."
                )
            syn = synthetic_severity_by_class
            if not syn:
                if n_cls <= 1:
                    syn = [0]
                else:
                    syn = [
                        min(int(round(i * (severity_classes - 1) / max(n_cls - 1, 1))), severity_classes - 1)
                        for i in range(n_cls)
                    ]
                logger.info("Auto synthetic_severity_by_class (per class id 0..n-1): %s", syn)
            if len(syn) < n_cls:
                raise ValueError(f"synthetic_severity_by_class length {len(syn)} < num classes {n_cls}")
            severity_arr = np.array([syn[int(lab)] for lab in labels], dtype=np.int64)
            severity_list = severity_arr.tolist()

    valid_indices = [i for i, p in enumerate(image_paths) if os.path.exists(p)]
    image_paths = [image_paths[i] for i in valid_indices]
    labels = labels[valid_indices]
    if severity_list is not None:
        severity_arr = np.array(severity_list, dtype=np.int64)[valid_indices]
        severity_list = severity_arr.tolist()

    logger.info("Found %s valid images", len(image_paths))

    mean, std = get_image_statistics(data_dir)

    tabular_matrix: Optional[np.ndarray] = None
    if tabular_dim and tabular_dim > 0:
        rows = []
        for p in image_paths:
            rec = meta_by_basename.get(Path(p).name) if meta_by_basename else None
            rows.append(_build_tabular_vector(rec, tabular_dim, tabular_extra_columns))
        tabular_matrix = np.stack(rows, axis=0)
        if not meta_by_basename:
            logger.warning(
                "model.tabular_dim=%s but no metadata_csv matched; using zero tabular vectors.",
                tabular_dim,
            )

    total_split = train_split + val_split + test_split
    if total_split <= 0:
        raise ValueError("train_split + val_split + test_split must be > 0")
    n = len(image_paths)
    train_n = int(n * (train_split / total_split))
    val_n = int(n * (val_split / total_split))
    test_n = n - train_n - val_n

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    tr_idx = perm[:train_n]
    va_idx = perm[train_n : train_n + val_n]
    te_idx = perm[train_n + val_n :]

    def subset(paths: List[str], lab: np.ndarray, sev: Optional[List[int]], idxs: List[int]) -> Tuple[List[str], List[int], Optional[List[int]]]:
        sp = [paths[i] for i in idxs]
        sl = [int(lab[i]) for i in idxs]
        sv = [sev[i] for i in idxs] if sev is not None else None
        return sp, sl, sv

    tr_paths, tr_labels, tr_sev = subset(image_paths, labels, severity_list, tr_idx)
    va_paths, va_labels, va_sev = subset(image_paths, labels, severity_list, va_idx)
    te_paths, te_labels, te_sev = subset(image_paths, labels, severity_list, te_idx)

    tr_tab = tabular_matrix[np.array(tr_idx, dtype=np.int64)] if tabular_matrix is not None else None
    va_tab = tabular_matrix[np.array(va_idx, dtype=np.int64)] if tabular_matrix is not None else None
    te_tab = tabular_matrix[np.array(te_idx, dtype=np.int64)] if tabular_matrix is not None else None

    train_ds = XrayDataset(
        tr_paths,
        tr_labels,
        tr_sev,
        tabular_vectors=tr_tab,
        image_size=image_size,
        augment=augment_train,
        mean=mean,
        std=std,
    )
    val_ds = XrayDataset(
        va_paths,
        va_labels,
        va_sev,
        tabular_vectors=va_tab,
        image_size=image_size,
        augment=False,
        mean=mean,
        std=std,
    )
    test_ds = (
        XrayDataset(
            te_paths,
            te_labels,
            te_sev,
            tabular_vectors=te_tab,
            image_size=image_size,
            augment=False,
            mean=mean,
            std=std,
        )
        if test_n > 0
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader
