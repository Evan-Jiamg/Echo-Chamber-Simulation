import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
from numeric_model import NumericWorld


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="scale_free",
                        choices=["small_world", "scale_free", "random"],
                        help="網路結構類型")
    parser.add_argument("--update_method", default="degroot",
                        choices=["degroot", "fj", "bcm"],
                        help="Opinion Update 方法")
    parser.add_argument("--num_agents", default=50, type=int,
                        help="Agent 數量")
    parser.add_argument("--step_count", default=30, type=int,
                        help="模擬步數")
    parser.add_argument("--seed", default=50, type=int,
                        help="Random seed")
    parser.add_argument("--epsilon", default=0.5, type=float,
                        help="BCM 的 epsilon 閾值")
    parser.add_argument("--run_all", action="store_true",
                        help="一次跑完所有 3 網路 x 3 方法共 9 組實驗")

    args = parser.parse_args()
    set_seed(args.seed)

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments_numeric')

    if args.run_all:
        network_types = ["scale_free", "small_world", "random"]
        update_methods = ["degroot", "fj", "bcm"]

        for network_type in network_types:
            for update_method in update_methods:
                print(f"\n{'='*50}")
                print(f"Network: {network_type} | Method: {update_method}")
                print(f"{'='*50}")
                set_seed(args.seed)
                model = NumericWorld(
                    network_type=network_type,
                    update_method=update_method,
                    num_agents=args.num_agents,
                    seed=args.seed,
                    epsilon=args.epsilon,
                    exp_dir=exp_dir
                )
                model.run_model(args.step_count)
    else:
        print(f"\n{'='*50}")
        print(f"Network: {args.network_type} | Method: {args.update_method}")
        print(f"{'='*50}")
        model = NumericWorld(
            network_type=args.network_type,
            update_method=args.update_method,
            num_agents=args.num_agents,
            seed=args.seed,
            epsilon=args.epsilon,
            exp_dir=exp_dir
        )
        model.run_model(args.step_count)
