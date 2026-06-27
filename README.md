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

## Project structure

```
qklstm/
├── quantum_kernel.py   # ZZ-feature-map quantum kernel: K_Q(x1,x2)=|<φ(x1)|φ(x2)>|²
├── qklstm_model.py     # QuantumKernelLSTM: quantum-kernel fusion + LSTM + dense head
├── qaoa.py             # QAOA MaxCut solver (circuit, expectation, COBYLA optimize)
├── optimizer.py        # QKLSTMQAOAOptimizer: encode / generate / train / predict / solve
└── visualization.py    # 4-panel results figure (training, params, comparison, kernel)

examples/demo_maxcut.py # end-to-end demonstration with CLI flags
tests/test_smoke.py     # fast tests for both quantum and classical paths
docs/MATH.md            # step-by-step mathematical deep-dive
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

- Scale to larger problems (>10 qubits) and deeper QAOA.
- Other problem families (TSP, portfolio optimization).
- Alternative quantum-kernel feature maps.
- Deployment on real quantum hardware via IBM Quantum.

## License

Released under the terms in [LICENSE](LICENSE).