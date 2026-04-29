import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
from dynamic_hybrid_model import HybridDynamicWorld


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="scale_free",
                        choices=["scale_free", "small_world", "random"])
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="0=pure LLM, 1=pure Numeric")
    parser.add_argument("--sigma", default=0.5, type=float)
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--step_count", default=30, type=int)
    parser.add_argument("--seed", default=50, type=int)
    parser.add_argument("--topic", default="euthanasia")
    parser.add_argument("--gpt_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--temp", default=0.5, type=float)
    parser.add_argument("--no_long_memory", action="store_true")
    parser.add_argument("--run_all_alpha", action="store_true",
                        help="Scan alpha in {0.0, 0.1, 0.2, ..., 1.0}")
    parser.add_argument("--run_all_networks", action="store_true",
                        help="Run scale_free, small_world, random")

    args = parser.parse_args()

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments_hybrid')
    belief_keywords_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')
    leaders = [10, 30]

    network_types = (["scale_free", "small_world", "random"]
                     if args.run_all_networks else [args.network_type])
    alpha_list = ([round(a * 0.1, 1) for a in range(11)]
                  if args.run_all_alpha else [args.alpha])

    for network_type in network_types:
        for alpha in alpha_list:
            print(f"\n{'='*55}")
            print(f"Network: {network_type} | K={args.K} | alpha={alpha}")
            print(f"{'='*55}")
            set_seed(args.seed)
            try:
                model = HybridDynamicWorld(
                    network_type=network_type,
                    K=args.K,
                    alpha=alpha,
                    sigma=args.sigma,
                    num_agents=args.num_agents,
                    seed=args.seed,
                    topic=args.topic,
                    gpt_model=args.gpt_model,
                    belief_keywords_file=belief_keywords_file,
                    exp_dir=exp_dir,
                    leaders=leaders,
                    temp=args.temp,
                    with_long_memory=not args.no_long_memory,
                )
                model.run_model(args.step_count)
            except FileNotFoundError as e:
                print(f"[Skipped] alpha={alpha} requires LLM backgrounds.\n  {e}")
