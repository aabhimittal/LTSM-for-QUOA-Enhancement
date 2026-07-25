"""Cross-size transfer demo: train on small graphs, predict for larger ones.

The original single-size optimiser cannot do this at all -- its input dimension
is ``n_qubits + 2``, so a model trained on 5-node graphs has the wrong tensor
shape for a 9-node graph.  With the size-agnostic
:class:`~qklstm.encoding.SpectralEncoder`, one model covers every size.

Usage
-----
    python examples/demo_cross_size.py             # train {4,5,6} -> test {4..9}
    python examples/demo_cross_size.py --quick     # tiny/fast configuration
    python examples/demo_cross_size.py --no-quantum

Results are reported as **approximation ratios** (expected cut / exact optimum),
which are comparable across sizes -- raw cut values are not, since larger graphs
simply have more edges to cut.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qklstm import CrossSizeQKLSTMOptimizer, SpectralEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-size QK-LSTM transfer demo")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--problems-per-size", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-problems", type=int, default=5)
    parser.add_argument("--qaoa-depth", type=int, default=1)
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-size-hint", action="store_true",
                        help="strictly size-blind encoding")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.train_sizes = [4, 5]
        args.test_sizes = [4, 5, 6, 7]
        args.problems_per_size, args.epochs, args.eval_problems = 6, 10, 3

    use_quantum = not args.no_quantum
    encoder = SpectralEncoder(include_size_hint=not args.no_size_hint)

    print("\n" + "=" * 64)
    print(" " * 12 + "CROSS-SIZE QK-LSTM TRANSFER DEMO")
    print("=" * 64)
    print("\nConfiguration:")
    print(f"  - Train sizes:    {args.train_sizes}")
    print(f"  - Test sizes:     {args.test_sizes}")
    print(f"  - QAOA depth:     {args.qaoa_depth}")
    print(f"  - Encoding dim:   {encoder.dim}  (constant for every graph size)")
    print(f"  - Quantum kernel: {'ENABLED' if use_quantum else 'DISABLED'}")

    optimizer = CrossSizeQKLSTMOptimizer(
        train_sizes=args.train_sizes,
        qaoa_depth=args.qaoa_depth,
        encoder=encoder,
        use_quantum=use_quantum,
        lstm_hidden=64,
    )

    optimizer.train(
        n_problems_per_size=args.problems_per_size,
        n_epochs=args.epochs,
        batch_size=8,
    )

    results = optimizer.evaluate_transfer(
        test_sizes=args.test_sizes, n_problems=args.eval_problems
    )

    unseen = [n for n in args.test_sizes if n not in optimizer.train_sizes]
    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    print("\n'predicted' uses a single forward pass -- no circuit optimisation at all.")
    print("'refined' warm-starts a short QAOA run from that prediction.")
    if unseen:
        print(f"\nSizes {unseen} were never seen during training, yet the same")
        print("model produces parameters for them: the encoding has the same")
        print("length for every graph, so nothing needs retraining.")
        for n in unseen:
            r = results[n]
            print(f"  n={n}: random {r['random']:.3f} -> "
                  f"predicted {r['predicted']:.3f} -> refined {r['refined']:.3f}")

    print("\n" + "=" * 64)
    print("DEMONSTRATION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
