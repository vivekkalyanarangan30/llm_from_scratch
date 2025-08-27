# Repository layout (Part 3)
#
#   part_3/
#     orchestrator.py              # runs tests + a small generation demo
#     tokenizer.py                 # local byte-level tokenizer (self-contained)
#     rmsnorm.py                   # 3.1 RMSNorm
#     rope.py                      # 3.2 RoPE cache + apply
#     swiglu.py                    # 3.3 SwiGLU FFN
#     kv_cache.py                  # 3.4/3.6 KV cache + rolling buffer
#     attn_modern.py               # attention w/ RoPE, sliding window, sink, optional KV cache
#     block_modern.py              # block = (RMSNorm|LN) + modern attention + (SwiGLU|GELU)
#     model_modern.py              # GPTModern wrapper with feature flags
#     demo_generate.py             # simple generation demo (shows KV cache + sliding window)
#     tests/
#       test_rmsnorm.py
#       test_rope_apply.py
#       test_kvcache_shapes.py
#
# Run from inside `part_3/`:
#   cd part_3
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
    p.add_argument("--demo", action="store_true", help="run a tiny generation demo")
    args = p.parse_args()

    # 1) run unit tests
    run("python -m pytest -q tests/test_rmsnorm.py")
    run("python -m pytest -q tests/test_rope_apply.py")
    run("python -m pytest -q tests/test_kvcache_shapes.py")

    # 2) (optional) generation demo
    if args.demo:
        run("python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    print("\nPart 3 checks complete. ✅")