"""生成生物医学指令微调数据集（MiniMind LoRA 实验用）。

运行::

    uv run python scripts/generate_biomed_instruct.py --n_examples 800 --seed 42

产物（写入 data/）:
  - biomed_instruct.jsonl  完整数据集
  - train.jsonl / val.jsonl / test.jsonl  8:1:1 划分

统计信息打印到 stdout。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接 `python scripts/xxx.py` 运行时，Python 只把 scripts/ 加入 sys.path，
# 导致 `from src...` 找不到模块；把项目根目录前置，保证任意目录下都能跑。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.data import build_dataset, split_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="生成生物医学指令微调数据集")
    parser.add_argument("--n_examples", type=int, default=800, help="目标样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out_dir", default="data", help="输出目录")
    args = parser.parse_args()

    records = build_dataset(n_examples=args.n_examples, seed=args.seed)
    splits = split_dataset(records, seed=args.seed)

    write_jsonl(f"{args.out_dir}/biomed_instruct.jsonl", records)
    for name in ("train", "val", "test"):
        write_jsonl(f"{args.out_dir}/{name}.jsonl", splits[name])

    print(f"generated {len(records)} examples -> {args.out_dir}/")
    for name, subset in splits.items():
        print(f"  {name:5s}: {len(subset)}")
    print(f"  ratio : {len(splits['train'])/len(records):.1f} / "
          f"{len(splits['val'])/len(records):.1f} / {len(splits['test'])/len(records):.1f}")


if __name__ == "__main__":
    main()
