import os
import argparse
import random
import numpy as np
from model import HybridDynamicWorld


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
    parser.add_argument("--num_agents", default=50, type=int)
    parser.add_argument("--step_count", default=None, type=int,
                        help="upper bound only; leave unset so the T_max in "
                             "core/convergence.py applies")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--topic", default="gun_control")
    parser.add_argument("--gpt_model", default="phi4")
    parser.add_argument("--backgrounds_label", default="phi4",
                        help="Label used to locate the agent backgrounds JSON file in data/")
    parser.add_argument("--temp", default=0.0, type=float)
    parser.add_argument("--scorer", default="roberta", choices=["roberta", "self_report"],
                        help="continuous stance scorer for Type-L opinions")
    parser.add_argument("--scorer_ckpt",
                        default=os.path.join(os.path.dirname(__file__), "..", "..",
                                             "Reddit-Dataset", "model", "final_model_bws.pt"),
                        help="Stage-1 regressor checkpoint (scorer=roberta)")
    parser.add_argument("--scorer_device", default="cuda")
    parser.add_argument("--exp_dir", default=None,
                        help="output root; defaults to <project>/experiments. "
                             "Point at /mnt/NewSSD for full sweeps (G-7).")
    parser.add_argument("--field_order", default="cot_first",
                        choices=["cot_first", "value_first"],
                        help="cot_first = reasoning derives the position (default)")
    parser.add_argument("--no_long_memory", action="store_true")
    parser.add_argument("--run_all_alpha", action="store_true",
                        help="Scan alpha in {0.0, 0.1, 0.2, ..., 1.0}")
    parser.add_argument("--run_all_networks", action="store_true",
                        help="Run scale_free, small_world, random")

    args = parser.parse_args()

    exp_dir = args.exp_dir or os.path.join(
        os.path.dirname(__file__), '..', 'experiments')
    belief_keywords_file = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'lexicons',
        'belief_keywords.json')
    n = args.num_agents
    leaders = [n // 5, n * 3 // 5]  # scale with num_agents (e.g. 50→[10,30], 30→[6,18])

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
                    num_agents=args.num_agents,
                    seed=args.seed,
                    topic=args.topic,
                    gpt_model=args.gpt_model,
                    belief_keywords_file=belief_keywords_file,
                    exp_dir=exp_dir,
                    leaders=leaders,
                    temp=args.temp,
                    with_long_memory=not args.no_long_memory,
                    backgrounds_label=args.backgrounds_label,
                    field_order=args.field_order,
                    scorer_kind=args.scorer,
                    scorer_kwargs=(
                        dict(ckpt_path=os.path.abspath(args.scorer_ckpt),
                             device=args.scorer_device)
                        if args.scorer == "roberta" else {}
                    ),
                )
                model.run_model(args.step_count)
            except FileNotFoundError as e:
                print(f"[Skipped] alpha={alpha} requires LLM backgrounds.\n  {e}")
