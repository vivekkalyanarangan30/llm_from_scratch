# Repository layout (Part 1)
#
#   part_1/
#     orchestrator.py               # runs demos/tests/visualizations for Part 1
#     pos_encoding.py               # 1.1 positional encodings (learned + sinusoidal)
#     attn_numpy_demo.py            # 1.2 self-attention math with tiny numbers (NumPy)
#     single_head.py                # 1.3 single attention head (PyTorch)
#     multi_head.py                 # 1.4 multi-head attention (with shape tracing)
#     ffn.py                        # 1.5 feed-forward network (GELU, width = mult*d_model)
#     block.py                      # 1.6 Transformer block (residuals + LayerNorm)
#     attn_mask.py                  # causal mask helpers
#     vis_utils.py                  # plotting helpers (matrices & attention maps)
#     demo_mha_shapes.py            # prints explicit matrix multiplications & shapes step-by-step
#     demo_visualize_multi_head.py  # saves attention heatmaps per head (grid)
#     out/                          # (created at runtime) images & logs live here
#     tests/
#       test_attn_math.py           # correctness: tiny example vs PyTorch single-head
#       test_causal_mask.py         # verifies masking behavior
#
# NOTE ON IMPORTS
# ----------------
# All imports are LOCAL. Run from inside `part_1/`.
# Example quickstart (CPU ok):
#   cd part_1
#   python orchestrator.py --visualize


import subprocess, sys, pathlib, argparse, shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"


def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help="run visualization scripts and save PNGs to ./out")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)

    # 1.2 sanity check: NumPy tiny example
    run("python attn_numpy_demo.py")

    # 1.3/1.4 unit tests
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_causal_mask.py")

    # Matrix math walkthrough for MHA
    run("python demo_mha_shapes.py")

    if args.visualize:
        run("python demo_visualize_multi_head.py")
        print(f"\nVisualization images saved to: {OUT}")

    print("\nAll Part 1 demos/tests completed. ✅")


if __name__ == "__main__":
    main()