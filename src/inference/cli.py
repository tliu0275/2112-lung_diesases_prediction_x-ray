"""命令行入口：直接传入一张或多张图片路径即可检测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import PneumoniaPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="肺炎类型 + 严重程度分析（输入图片路径即可）",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="一张或多张图像路径（jpg/png/dcm 等）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="训练/推理配置文件",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="覆盖默认 checkpoints/best_model.pth")
    parser.add_argument("--device", type=str, default=None, help="cpu 或 cuda")
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON，便于程序解析",
    )
    parser.add_argument(
        "--no-dataset-norm",
        action="store_true",
        help="不用数据集统计量做归一化（更快；可能与训练不一致）",
    )
    parser.add_argument("--age", type=float, default=None, help="患者年龄（与 model.tabular_dim 联用）")
    parser.add_argument("--gender", type=str, default=None, help="M / F（与 model.tabular_dim 联用）")
    args = parser.parse_args()

    predictor = PneumoniaPredictor.from_config_file(
        args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_dataset_normalization=not args.no_dataset_norm,
    )

    for img in args.images:
        result = predictor.predict(img, age=args.age, gender=args.gender)
        if args.json:
            print(PneumoniaPredictor.to_json(result))
        else:
            print(PneumoniaPredictor.format_report(result))
            print()


if __name__ == "__main__":
    main()
