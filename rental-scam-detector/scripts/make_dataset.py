"""Генерирует синтетический размеченный датасет признаков (по умолчанию data/dataset.csv).

Пример: uv run python scripts/make_dataset.py --n 2000 --scam-ratio 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scam_detector.config import DATA_DIR
from scam_detector.synth import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="число объявлений")
    ap.add_argument("--scam-ratio", type=float, default=0.25, help="доля мошеннических")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "dataset.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = make_dataset(args.n, args.scam_ratio, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {len(df)} строк, scam={df['label'].mean():.1%}")


if __name__ == "__main__":
    main()
