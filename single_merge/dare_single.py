"""
Merge a single RL checkpoint back into its base using dare_linear.

Formula per tensor:
  result = base + weight * L1Rescale(delta * Bernoulli(density))
  where delta = rl_checkpoint - base

Usage:
  python longcat_merge/dare_single.py \
    --base /path/to/sft_dpo_base \
    --rl /path/to/math_rl_checkpoint \
    --out /path/to/merged_output \
    --weight 0.7 \
    --density 0.5 \
    --dtype bfloat16
"""

import argparse

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


def main():
    p = argparse.ArgumentParser(description="DARE-linear single-model merge")
    p.add_argument("--base", required=True, help="Base (SFT/DPO) model path")
    p.add_argument("--rl", required=True, help="RL checkpoint path")
    p.add_argument("--out", required=True, help="Output path")
    p.add_argument("--weight", type=float, default=0.7)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--lazy-unpickle", action="store_true")
    args = p.parse_args()

    # dare_linear defaults: rescale=True, normalize=False, consensus=None
    config = MergeConfiguration(
        merge_method="dare_linear",
        base_model=args.base,
        models=[
            InputModelDefinition(
                model=args.rl,
                parameters={"weight": args.weight, "density": args.density},
            ),
        ],
        dtype=args.dtype,
    )

    options = MergeOptions(
        cuda=args.cuda,
        lazy_unpickle=args.lazy_unpickle,
    )

    print(f"dare_linear merge: weight={args.weight}, density={args.density}")
    print(f"  base:   {args.base}")
    print(f"  rl:     {args.rl}")
    print(f"  output: {args.out}")

    run_merge(config, args.out, options=options)
    print("Done.")


if __name__ == "__main__":
    main()
