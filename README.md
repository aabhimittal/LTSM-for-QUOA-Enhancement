# QK-LSTM for QAOA Enhancement

> A hybrid **quantum-classical** framework that uses a **Quantum Kernel-based
> LSTM (QK-LSTM)** to predict good **QAOA** parameters for combinatorial
> optimization — turning a slow per-problem optimization into a fast,
> single-forward-pass prediction.

*(The repository name uses "QUOA"; the algorithm is the Quantum Approximate
Optimization Algorithm, **QAOA**.)*

---

## Why this project?

Solving a combinatorial problem (here **MaxCut**) with QAOA means searching for
good variational angles `(γ, β)`. Done classically, that search costs ~100+
quantum-circuit evaluations **per problem**. QK-LSTM instead *learns to optimize*:
it trains once on many solved problems and then predicts good parameters for new
problems instantly.

Three ideas combined:

| Pillar | Role | Analogy |
| --- | --- | --- |
| **LSTM** | Learns temporal patterns from optimization trajectories. | A grandmaster recognizing positions instantly. |
| **Quantum Kernel** | Measures similarity in an exponentially large Hilbert space. | Viewing the city from a helicopter, not the street. |
| **QAOA** | The quantum optimizer being accelerated. | An orchestra searching for perfect harmony. |

---

## How it works

```
Problem graph ──► encode ──► [Quantum Kernel fusion] ──► LSTM ──► (γ, β) prediction
                                                                        │
                              ┌─────────────────────────────────────────┤
                              ▼                       ▼                   ▼
                        Baseline QAOA          QK-LSTM only        QK-LSTM + refine
                     (full optimization)     (1 forward pass)     (warm-started QAOA)
```

1. **Encode** each MaxCut graph as `[degree sequence, edge density, #edges]`.
2. **Quantum-kernel fusion** (ZZ-feature map) compares the input to a reference
   set and fuses the similarity vector back in via a residual:
   `x̃ = x + 0.5 · proj(K_Q(x, X_ref))`.
3. **LSTM** consumes the trajectory and predicts the QAOA angles `(γ, β)`.
4. **Evaluate** the prediction directly, and optionally **refine** it with a few
   classical QAOA steps (warm start).

The quantum kernel is treated as **fixed preprocessing**: gradients flow through
the projection and LSTM, not through the (non-differentiable) quantum circuit.

---

## Installation

```bash
git clone https://github.com/aabhimittal/ltsm-for-quoa-enhancement.git
cd ltsm-for-quoa-enhancement

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. The quantum backend uses **Qiskit** + **qiskit-aer**.

---

## Quick start

```python
from qklstm import QKLSTMQAOAOptimizer

opt = QKLSTMQAOAOptimizer(n_qubits=6, qaoa_depth=2, use_quantum=True)
opt.train(n_problems=50, n_epochs=30)

problem = [(0, 1), (1, 2), (2, 3), (3, 0)]      # a 4-cycle (square)
results = opt.solve_with_prediction(problem)
print(results["predicted"]["value"], results["refined"]["value"])
```

Run the full demonstration (trains, compares, and saves a figure):

```bash
python examples/demo_maxcut.py            # full demo  -> examples/results.png
python examples/demo_maxcut.py --quick    # fast smoke run
python examples/demo_maxcut.py --no-quantum   # classical LSTM ablation
```

---

## Cross-size transfer: train small, predict large

The single-size optimizer encodes a graph as `[degree_0 … degree_{n-1}, density,
n_edges]`, so `input_dim = n_qubits + 2`. That has two costs: the encoding is **not
permutation-invariant** (relabeling vertices changes the vector), and it is **welded to
one problem size** — a model trained on 6-node graphs has the wrong tensor shape for a
9-node graph. Every size needs retraining, which defeats the "train once, predict
instantly" premise.

`SpectralEncoder` fixes both. It describes a graph by its **normalized-Laplacian
spectrum** plus scale-free structural statistics:

- `L = I − D^{-1/2} A D^{-1/2}` has all eigenvalues in `[0, 2]` **for any graph size** —
  that bound is what makes spectra comparable across sizes (the combinatorial Laplacian's
  spectrum grows with degree, so it would not work).
- The spectrum is summarized by 11 fixed **quantiles**, giving constant length for any `n`.
- Degree moments are normalized by `n−1`; clustering, triangle density, assortativity and
  edge density are already scale-free.

Every feature is a function of the *unordered* spectrum or a normalized aggregate, so the
encoding is permutation-invariant by construction — verified to ~1e-17 in the tests.

```python
from qklstm import CrossSizeQKLSTMOptimizer

opt = CrossSizeQKLSTMOptimizer(train_sizes=(4, 5, 6), qaoa_depth=1)
opt.train(n_problems_per_size=15, n_epochs=40)

opt.evaluate_transfer(test_sizes=[4, 5, 6, 7, 8, 9])   # 7-9 never seen
```

```bash
python examples/demo_cross_size.py           # train {4,5,6} -> test {4..9}
python examples/demo_cross_size.py --quick   # fast run
```

### Measured results

Mean **approximation ratio** (expected cut / exact optimum, so values are comparable
across sizes), 5 random graphs per size, `p = 1`, classical kernel:

| nodes | in training? | random | QK-LSTM predicted | + refinement |
| ---: | :---: | ---: | ---: | ---: |
| 4 | yes | 0.580 | 0.794 | 0.820 |
| 5 | yes | 0.786 | 0.835 | 0.889 |
| 6 | yes | 0.661 | 0.816 | 0.820 |
| **7** | **NEW** | 0.625 | **0.788** | 0.791 |
| **8** | **NEW** | 0.553 | **0.784** | 0.788 |
| **9** | **NEW** | 0.550 | **0.785** | 0.790 |

Mean gain over random on unseen sizes: **+0.21**. Prediction quality stays essentially
flat (0.788 / 0.784 / 0.785) as graphs grow past the training range — the transfer does
not decay with size. "Predicted" is a **single forward pass with zero circuit
optimization**; refinement adds only ~0.005, meaning the warm start already lands close
to the optimum COBYLA converges to.

### Why this needed a symmetry fix

A first implementation scored **worse than random** (−0.08). The cause was not the
encoding but the *labels*: the QAOA objective is invariant under

| symmetry | consequence |
| --- | --- |
| `γ → γ + π` | integer cost spectrum |
| `β → β + π/2` | `X^⊗n` commutes with the cost and fixes `|+⟩^ⁿ` |
| `(γ, β) → (−γ, −β)` | time reversal (the state is real) |

so COBYLA returns optima scattered across equivalent branches (label std ≈ 1.8 rad).
Regressing with MSE onto a multimodal target pulls the model toward the *mean* of
several valid solutions, which is itself not a solution. Three changes fix it:

1. **`canonicalize_qaoa_params`** folds every label onto one representative of its
   symmetry orbit (`γ ∈ [0, π)`, `β ∈ [0, π/4]`). Label scatter drops to *exactly zero*.
2. **Multi-restart labelling** (`QAOA.optimize(n_restarts=…)`) so labels are near-global
   rather than whichever local optimum was found first.
3. **`PeriodicAngleLoss`** — `1 − cos(2π·Δ/period)` — which treats `γ = 0.01` and
   `γ = π − 0.01` as the near-identical circuits they are, instead of maximally distant.

All three symmetries are verified to machine precision in `tests/test_symmetry.py`.

---

## Project structure

```
qklstm/
├── quantum_kernel.py   # ZZ-feature-map quantum kernel: K_Q(x1,x2)=|<φ(x1)|φ(x2)>|²
├── qklstm_model.py     # QuantumKernelLSTM: quantum-kernel fusion + LSTM + dense head
├── qaoa.py             # QAOA MaxCut solver + symmetry canonicalization + exact optimum
├── encoding.py         # DegreeEncoder (original) and SpectralEncoder (invariant)
├── optimizer.py        # QKLSTMQAOAOptimizer: single-size meta-optimizer
├── cross_size.py       # CrossSizeQKLSTMOptimizer + PeriodicAngleLoss
└── visualization.py    # 4-panel results figure (training, params, comparison, kernel)

examples/demo_maxcut.py      # single-size end-to-end demonstration
examples/demo_cross_size.py  # train on small graphs, transfer to larger ones
tests/test_smoke.py          # core pipeline, quantum and classical paths
tests/test_encoding.py       # permutation invariance, size-agnosticism
tests/test_symmetry.py       # QAOA symmetries, canonicalization, periodic loss
tests/test_cross_size.py     # transfer end-to-end + backward compatibility
docs/MATH.md                 # step-by-step mathematical deep-dive
```

---

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite runs on tiny problems and covers the quantum kernel, QAOA, the model
forward pass (both `use_quantum=True/False`), and the full optimizer pipeline.

---

## Notes on faithfulness & fixes

This implementation is a faithful, modular port of the original QK-LSTM design.
A few correctness fixes were applied so the pipeline runs end-to-end:

- **Quantum projection dimension.** The original projection was
  `Linear(input_dim, input_dim)`, but quantum-kernel features have dimension
  `m_ref` (the reference-set size). It is now built lazily as
  `Linear(m_ref, input_dim)` once the reference set is known, so the residual
  fusion is dimensionally consistent (the pipeline previously crashed with
  `use_quantum=True`).
- **MaxCut bit ordering.** Statevector indices are mapped so that `bitstring[i]`
  corresponds to graph vertex `i`.
- **Random-graph generation.** The edge count is capped at `C(n, 2)`; otherwise
  small graphs could request more unique edges than exist, hanging the sampler in
  an infinite loop.
- **Robustness.** RNG seeding for reproducibility and guarded divisions in the
  performance comparison.

See [`docs/MATH.md`](docs/MATH.md) for the full mathematical treatment, from
qubits and superposition through quantum kernels, LSTM gates, and the end-to-end
training/inference pipeline.

---

## Roadmap

- Learned-optimizer mode: feed real optimization trajectories (γ, β, cost) through the
  LSTM so the recurrence models the *update rule* rather than a static encoding.
- Generalize the backend from unweighted MaxCut to arbitrary Ising/QUBO problems
  (weighted MaxCut, number partitioning, portfolio optimization).
- Classical-kernel ablation (RBF vs quantum) plus kernel caching, to quantify how much
  the quantum kernel actually contributes.
- Depth extrapolation with INTERP/FOURIER baselines (`p → p+1` parameter transfer).
- Deployment on real quantum hardware via IBM Quantum.

## License

Released under the terms in [LICENSE](LICENSE).