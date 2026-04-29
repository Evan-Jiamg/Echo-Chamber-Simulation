import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
from dynamic_llm_model import DynamicLLMWorld


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed set to {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="scale_free",
                        choices=["scale_free", "small_world", "random"])
    parser.add_argument("--K", default=5, type=int,
                        help="K-NN 的 K 值")
    parser.add_argument("--sigma", default=1.0, type=float,
                        help="Gaussian g 的 sigma（LLM belief -2~+2，預設 1.0）")
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--step_count", default=30, type=int)
    parser.add_argument("--seed", default=50, type=int)
    parser.add_argument("--topic", default="euthanasia")
    parser.add_argument("--gpt_model", default="gpt-4o-mini-2024-07-18",
                        help="OpenAI model ID")
    parser.add_argument("--temp", default=0.5, type=float)
    parser.add_argument("--no_long_memory", action="store_true",
                        help="停用長期記憶（ablation 用）")
    parser.add_argument("--run_all_K", action="store_true",
                        help="一次跑 K = 3, 5, 10, 20, 49")
    parser.add_argument("--run_all_networks", action="store_true",
                        help="一次跑 scale_free, small_world, random")

    args = parser.parse_args()

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments_dynamic_llm')
    belief_keywords_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')
    leaders = [10, 30]

    network_types = ["scale_free", "small_world", "random"] if args.run_all_networks else [args.network_type]
    K_list = [3, 5, 10, 20, 49] if args.run_all_K else [args.K]

    for network_type in network_types:
        for K in K_list:
            print(f"\n{'='*55}")
            print(f"Network: {network_type} | K = {K} | sigma = {args.sigma}")
            print(f"{'='*55}")
            set_seed(args.seed)
            model = DynamicLLMWorld(
                network_type=network_type,
                K=K,
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
