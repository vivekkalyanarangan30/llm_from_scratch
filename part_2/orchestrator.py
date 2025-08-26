# Repository layout (Part 2)
#
#   part_2/
#     orchestrator.py           # runs quick smoke-train + eval + sample
#     tokenizer.py              # 2.1 byte-level tokenizer (0..255)
#     dataset.py                # 2.2 dataset + batching + shift
#     utils.py                  # 2.5 sampling helpers (top-k/top-p)
#     model_gpt.py              # tiny GPT: tok/pos emb + blocks + head
#     train.py                  # 2.3/2.4 training loop w/ val eval & ckpt
#     sample.py                 # 2.5 text generation from a checkpoint
#     eval_loss.py              # 2.6 evaluate loss on a file/ckpt
#     tests/
#       test_tokenizer.py       # round-trip encode/decode
#       test_dataset_shift.py   # label shift sanity
#     runs/                     # (created at runtime) checkpoints & logs
#
# NOTE ON IMPORTS
# ----------------
# All imports are LOCAL. Run from inside `part_2/`.
# Example quickstart (CPU ok):
#   cd part_2
#   python train.py --data tiny.txt --steps 300 --sample_every 100
#   python sample.py --ckpt runs/min-gpt/model_best.pt --tokens 200 --prompt 'Once upon a time '


import subprocess, sys, pathlib, shlex

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / 'runs' / 'min-gpt'

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == '__main__':
    # quick smoke training on a tiny file path tiny_hi.txt; adjust as needed
    run("python train.py --data tiny_hi.txt --steps 400 --sample_every 100 --eval_interval 100 --batch_size 32 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128")

    # sample from the best checkpoint
    run(f"python sample.py --ckpt {RUNS}/model_best.pt --tokens 200 --prompt 'Once upon a time '")

    # evaluate final val loss
    run(f"python eval_loss.py --data tiny_hi.txt --ckpt {RUNS}/model_best.pt --iters 50 --block_size 128")