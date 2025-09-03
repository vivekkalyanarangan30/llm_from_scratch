# Repository layout (Part 7)
#
#   part_7/
#     orchestrator.py           # run unit tests + optional tiny RM demo
#     data_prefs.py             # 7.1 HF preference loader (+tiny fallback)
#     collator_rm.py            # pairwise tokenization → (pos, neg) tensors
#     model_reward.py           # 7.2 reward model (Transformer encoder → scalar)
#     loss_reward.py            # 7.3 Bradley–Terry & margin-ranking losses
#     train_rm.py               # minimal one‑GPU training on tiny slice
#     eval_rm.py                # 7.4 sanity checks & simple accuracy on val
#     tests/
#       test_bt_loss.py
#       test_reward_forward.py
#
# Run from inside `part_7/`:
#   cd part_7
#   python orchestrator.py --demo
#   pytest -q

import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny reward‑model demo")
    args = p.parse_args()

    # 1) unit tests
    run("python -m pytest -q tests/test_bt_loss.py")
    run("python -m pytest -q tests/test_reward_forward.py")

    # 2) optional demo: tiny train + eval
    if args.demo:
        run("python train_rm.py --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --loss bt --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split train[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 7 checks complete. ✅")