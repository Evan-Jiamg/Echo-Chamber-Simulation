import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
from dynamic_numeric_model import DynamicWorld


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="scale_free",
                        choices=["scale_free", "small_world", "random"],
                        help="網路類型")
    parser.add_argument("--K", default=5, type=int,
                        help="K-NN 的 K 值（每個 agent 的 out-neighbor 數量）")
    parser.add_argument("--sigma", default=0.5, type=float,
                        help="Gaussian g 的 sigma")
    parser.add_argument("--num_agents", default=50, type=int,
                        help="Agent 數量")
    parser.add_argument("--step_count", default=30, type=int,
                        help="模擬步數")
    parser.add_argument("--seed", default=50, type=int,
                        help="Random seed")
    parser.add_argument("--run_all_K", action="store_true",
                        help="一次跑完 K = 3, 5, 10, 20, 49 共 5 組實驗")
    parser.add_argument("--run_all_networks", action="store_true",
                        help="一次跑完 scale_free, small_world, random 三種網路")
    parser.add_argument("--update_method", default="degroot",
                        choices=["degroot", "fj", "bcm"],
                        help="Opinion update rule: degroot / fj / bcm")
    parser.add_argument("--epsilon", default=0.3, type=float,
                        help="BCM 的 epsilon（bounded confidence threshold）")
    parser.add_argument("--run_all_methods", action="store_true",
                        help="一次跑完 degroot / fj / bcm 三種更新方式")

    args = parser.parse_args()

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments_dynamic')

    network_types = ["scale_free", "small_world", "random"] if args.run_all_networks else [args.network_type]
    K_list = [3, 5, 10, 20, 49] if args.run_all_K else [args.K]
    methods = ["degroot", "fj", "bcm"] if args.run_all_methods else [args.update_method]

    for network_type in network_types:
        for K in K_list:
            for method in methods:
                print(f"\n{'='*55}")
                print(f"Network: {network_type} | K = {K} | method = {method} | sigma = {args.sigma}")
                print(f"{'='*55}")
                set_seed(args.seed)
                model = DynamicWorld(
                    network_type=network_type,
                    K=K,
                    sigma=args.sigma,
                    num_agents=args.num_agents,
                    seed=args.seed,
                    exp_dir=exp_dir,
                    update_method=method,
                    epsilon=args.epsilon,
                )
                model.run_model(args.step_count)
