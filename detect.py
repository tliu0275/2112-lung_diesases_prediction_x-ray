#!/usr/bin/env python3
"""
项目根目录一键检测入口（最短命令）:
  python detect.py yihan.jpg
  python detect.py a.jpg b.jpg --json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.cli import main

if __name__ == "__main__":
    main()
