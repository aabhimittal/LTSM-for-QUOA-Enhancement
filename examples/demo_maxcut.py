"""End-to-end demonstration of QK-LSTM enhanced QAOA on MaxCut.

Usage
-----
    python examples/demo_maxcut.py            # full demo
    python examples/demo_maxcut.py --quick    # tiny/fast configuration
    python examples/demo_maxcut.py --no-quantum   # classical LSTM ablation

The demo trains the QK-LSTM on randomly generated MaxCut problems, then compares
three strategies on a held-out 6-node test graph:

1. Baseline   - full classical QAOA optimisation.
2. QK-LSTM    - single forward-pass parameter prediction.
3. Refinement - the prediction used as a warm start for a few QAOA steps.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running the script directly from a fresh checkout.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qklstm import QKLSTMQAOAOptimizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QK-LSTM enhanced QAOA demo")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--qaoa-depth", type=int, default=2)
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-quantum", action="store_true",
                        help="disable the quantum kernel (classical LSTM only)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny configuration for a fast smoke run")
    parser.add_argument("--save-fig", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_qubits, args.qaoa_depth = 4, 1
        args.n_problems, args.epochs, args.batch_size = 8, 5, 4

    use_quantum = not args.no_quantum

    print("\n" + "=" * 70)
    print(" " * 16 + "QK-LSTM ENHANCED QAOA SYSTEM")
    print(" " * 12 + "Quantum-Classical Hybrid Optimization")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  - Problem size:      {args.n_qubits} qubits")
    print(f"  - QAOA depth:        {args.qaoa_depth} layers")
    print(f"  - Training problems: {args.n_problems}")
    print(f"  - Training epochs:   {args.epochs}")
    print(f"  - Quantum kernel:    {'ENABLED' if use_quantum else 'DISABLED'}")

    optimizer = QKLSTMQAOAOptimizer(
        n_qubits=args.n_qubits,
        qaoa_depth=args.qaoa_depth,
        lstm_hidden=128,
        use_quantum=use_quantum,
    )

    optimizer.train(
        n_problems=args.n_problems,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Held-out test problem: a 6-cycle with three chords (or a square in --quick).
    if args.n_qubits >= 6:
        test_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # cycle
            (0, 3), (1, 4), (2, 5),                          # chords
        ]
    else:
        test_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    results = optimizer.solve_with_prediction(test_edges)

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    base = results["baseline"]["value"] or 1e-9
    pred_pct = (results["predicted"]["value"] / base - 1) * 100
    ref_pct = (results["refined"]["value"] / base - 1) * 100
    print(f"\n1. Direct prediction reached {pred_pct:+.1f}% relative to baseline.")
    print("2. Prediction is a single forward pass vs. ~100 QAOA evaluations.")
    print(f"3. Prediction + refinement reached {ref_pct:+.1f}% relative to baseline.")

    try:
        from qklstm import visualize_results

        visualize_results(optimizer, save_path=args.save_fig)
    except Exception as exc:  # plotting is optional; never fail the demo on it
        print(f"\n(Visualization skipped: {exc})")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
