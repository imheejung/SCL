import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="")
parser.add_argument("--dataset", default="travel")
parser.add_argument("--mode", default="all", choices=["single", "ablation", "all"])
parser.add_argument("--config", default="")
parser.add_argument("--exp_name", default="")
parser.add_argument("--gpu", default="1")
args = parser.parse_args()

PYTHON = "/data1/heejung/envs/travel/bin/python"
SEEDS = [2020, 2021, 2022, 2023, 2024]

ABLATIONS = [
    ("configs/ablation/wo_struct.yaml", "wo_struct"),
    ("configs/ablation/wo_potential.yaml", "wo_potential"),
    ("configs/ablation/wo_semantic.yaml", "wo_semantic"),
]

ALL_EXPERIMENTS = [
    ("lightgcn", "", "lightgcn"),
    ("ncl", "", "ncl"),
    ("scl", "", "scl"),
    ("scl", "configs/ablation/wo_struct.yaml", "wo_struct"),
    ("scl", "configs/ablation/wo_potential.yaml", "wo_potential"),
    ("scl", "configs/ablation/wo_semantic.yaml", "wo_semantic"),
]

def run_one(model, dataset, seed, config="", exp_name=""):
    cmd = [
        PYTHON,
        "run.py",
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
    ]

    if config:
        cmd.extend(["--config", config])

    if exp_name:
        cmd.extend(["--exp_name", exp_name])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

if args.mode == "single":
    for seed in SEEDS:
        run_one(args.model, args.dataset, seed, args.config, args.exp_name)

elif args.mode == "ablation":
    for config_path, exp_name in ABLATIONS:
        for seed in SEEDS:
            run_one(args.model, args.dataset, seed, config_path, exp_name)

elif args.mode == "all":
    for model, config_path, exp_name in ALL_EXPERIMENTS:
        for seed in SEEDS:
            run_one(model, args.dataset, seed, config_path, exp_name)