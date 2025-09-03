# Repository layout (Part 6)
#
#   part_6/
#     orchestrator.py           # run unit tests + optional tiny SFT demo
#     formatters.py             # 6.1 prompt/response templates
#     dataset_sft.py            # HF dataset loader (+tiny fallback) → (prompt, response)
#     collator_sft.py           # 6.2 causal LM labels with masking
#     curriculum.py             # 6.3 length‑based curriculum sampler
#     evaluate.py               # 6.4 simple exact/F1 metrics
#     train_sft.py              # minimal one‑GPU SFT loop (few steps)
#     sample_sft.py             # load ckpt & generate from instructions
#     tests/
#       test_formatter.py
#       test_masking.py
#
# Run from inside `part_6/`:
#   cd part_6
#   python orchestrator.py --demo
#   pytest -q

#
#   part_6/
#     orchestrator.py           # run unit tests + optional tiny SFT demo
#     formatters.py             # 6.1 prompt/response templates
#     dataset_sft.py            # HF dataset loader (+tiny fallback) → (prompt, response)
#     collator_sft.py           # 6.2 causal LM labels with masking
#     curriculum.py             # 6.3 length‑based curriculum sampler
#     evaluate.py               # 6.4 simple exact/F1 metrics
#     train_sft.py              # minimal one‑GPU SFT loop (few steps)
#     sample_sft.py             # load ckpt & generate from instructions
#     tests/
#       test_formatter.py
#       test_masking.py
#
# Run from inside `part_6/`:
#   cd part_6
#   python orchestrator.py --demo
#   pytest -q

### FILE: part_6/orchestrator.py
import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny SFT demo on a few samples")
    args = p.parse_args()

    # 1) unit tests
    run("python -m pytest -q tests/test_formatter.py")
    run("python -m pytest -q tests/test_masking.py")

    # 2) optional demo
    if args.demo:
        # --ckpt ../part_4/runs/part4-demo/model_last.pt # assumes Part 4 demo has been run
        run("python train_sft.py --data huggingface --ckpt ../part_4/runs/part4-demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What are the three primary colors?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What does DNA stand for?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'Reverse engineer this code to create a new version\ndef factorialize(num):\n  factorial = 1\n  for i in range(1, num):\n    factorial *= i\n  \n  return factorial' --tokens 64 --temperature 0.2")

    print("\nPart 6 checks complete. ✅")