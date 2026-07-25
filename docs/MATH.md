# QK-LSTM for QAOA — Mathematical Deep Dive

This document walks through the mathematics behind the implementation, from
quantum-mechanics basics to the end-to-end training and inference pipeline. It
maps directly onto the code in [`../qklstm`](../qklstm).

---

## 1. Quantum-mechanics foundations

### 1.1 States and Hilbert space

A quantum state is a unit vector (a "ket") `|ψ⟩`. For a single qubit:

```
|ψ⟩ = α|0⟩ + β|1⟩,   α, β ∈ ℂ,   |α|² + |β|² = 1
```

`|α|²` and `|β|²` are the probabilities of measuring `0` and `1`. For **n**
qubits the state lives in a `2ⁿ`-dimensional space (the tensor product of single
qubit spaces) — exponential growth is the root of "quantum advantage."

### 1.2 Inner product

```
⟨φ|ψ⟩ = Σᵢ φᵢ* ψᵢ
```

`|⟨φ|ψ⟩|²` is the overlap (similarity): `1` ⇒ identical, `0` ⇒ orthogonal.

---

## 2. QAOA — `qklstm/qaoa.py`

### 2.1 MaxCut

Given a graph `G = (V, E)`, partition `V` into two sets to maximize the number
of edges crossing the partition. With spins `zᵢ ∈ {−1, +1}`:

```
C(z) = Σ_{(i,j)∈E} (1 − zᵢ zⱼ) / 2
```

An edge contributes `1` when its endpoints differ (cut) and `0` otherwise.

### 2.2 Cost & mixer Hamiltonians

```
H_C = Σ_{(i,j)∈E} (I − Zᵢ Zⱼ)/2        (encodes the cut value)
H_M = Σᵢ Xᵢ                            (drives transitions between solutions)
```

`Z` is the Pauli-Z (eigenvalues ±1 on `|0⟩, |1⟩`); `X` is the bit-flip.

### 2.3 The QAOA state

Start from a uniform superposition `|s⟩ = |+⟩^{⊗n}` and alternate evolutions:

```
U_C(γ) = e^{−iγ H_C},   U_M(β) = e^{−iβ H_M}
|ψ(γ, β)⟩ = U_M(β_p) U_C(γ_p) … U_M(β_1) U_C(γ_1) |s⟩
```

In code this is `QAOA.create_qaoa_circuit` (`rzz` for the cost layer, `rx` for
the mixer).

### 2.4 Objective

We maximize the expected cut value

```
F(γ, β) = ⟨ψ(γ, β)| H_C |ψ(γ, β)⟩ = Σ_z C(z) · |⟨z|ψ(γ, β)⟩|²
```

`QAOA.compute_expectation` enumerates basis states, weights each cut by its
probability, and sums. `QAOA.optimize` maximizes `F` with COBYLA (scipy
minimizes `−F`).

---

## 3. Quantum kernels — `qklstm/quantum_kernel.py`

### 3.1 From classical to quantum kernels

A kernel measures similarity. Classical examples: linear `x·y`, RBF
`e^{−γ‖x−y‖²}`. A quantum feature map sends data to a quantum state:

```
x ↦ |φ(x)⟩ = U_φ(x) |0…0⟩
```

and the **quantum kernel** is the squared overlap (a real similarity in `[0,1]`):

```
K_Q(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
```

### 3.2 ZZ-feature map (used here)

Each layer applies, for every qubit `i` and neighbouring pair `(i, i+1)`:

```
1. Hadamard:    H            (superposition)
2. Encoding:    RZ(2 xᵢ)     (data rotation)
3. Entangling:  CX · RZ(2(π − xᵢ)(π − x_{i+1})) · CX   (ZZ correlations)
```

For `n` qubits the feature space is `2ⁿ`-dimensional while storage/computation
stay polynomial — the source of the exponential representational advantage.
`QuantumKernel.feature_map` builds the circuit; `compute_kernel` evaluates the
overlap via exact statevectors.

---

## 4. LSTM — `qklstm/qklstm_model.py`

An LSTM controls information flow through gates (`σ` is the sigmoid, `⊙` the
element-wise product):

```
fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)          forget gate
iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)          input gate
C̃ₜ = tanh(W_C·[hₜ₋₁, xₜ] + b_C)       candidate cell
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ              cell update
oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)          output gate
hₜ = oₜ ⊙ tanh(Cₜ)                     hidden state
```

The cell state `Cₜ` is the long-term memory; because `∂Cₜ/∂Cₜ₋₁ = fₜ`, the LSTM
avoids vanishing gradients when memory should persist. We use `torch.nn.LSTM`
and read the final hidden state into a small dense head producing the QAOA
parameters.

---

## 5. QK-LSTM integration — `qklstm/qklstm_model.py`

### 5.1 Quantum enhancement of the input

For each time-step vector `xₜ`, compute kernels against a reference set
`X_ref = {x₁, …, x_{m}}`:

```
Kₜ = [ K_Q(xₜ, x₁), …, K_Q(xₜ, x_m) ] ∈ ℝ^{m}
```

### 5.2 Projection and residual fusion

Because `Kₜ ∈ ℝ^{m}` and `xₜ ∈ ℝ^{d}` generally differ in size, project and fuse
as a residual:

```
x̃ₜ = xₜ + α · W_proj Kₜ,    W_proj ∈ ℝ^{d×m},   α = 0.5
```

> **Implementation fix.** `W_proj` is `Linear(m, d)` (not `Linear(d, d)`), built
> lazily in `set_reference_data` once `m = m_ref` is known. The original
> prototype used `Linear(d, d)` and crashed whenever `use_quantum=True`.

### 5.3 Gradients

The quantum kernel is **fixed preprocessing**: `Kₜ` is computed in the forward
pass and treated as constant, so gradients flow only through `W_proj` and the
LSTM:

```
∂L/∂W_proj = (∂L/∂x̃ₜ) · α · Kₜᵀ
```

(A parameter-shift rule could make the circuit differentiable; not needed here.)

---

## 6. End-to-end pipeline — `qklstm/optimizer.py`

### Training

For a dataset `D = {(G_n, θ_n*)}` of graphs and their optimal QAOA parameters:

```
1. Encode:   xₙ = [degree sequence, edge density, #edges]   (length n+2)
2. Kernels:  Kₙ = [K_Q(xₜ, X_ref)]ₜ
3. Predict:  θ̂ₙ = f_QK-LSTM(Xₙ, Kₙ; W)
4. Loss:     L = (1/N) Σ ‖θ̂ₙ − θ_n*‖²        (MSE)
5. Update:   W ← Adam(W, ∇_W L)
```

`generate_training_data` produces labels by solving random graphs with classical
QAOA; `train` runs the loop above (Adam, gradient clipping, train/val split).

### Inference

```
θ̂_new = f_QK-LSTM(X_new, K_new; W*)
γ = θ̂[0:p],   β = θ̂[p:2p]
run QAOA(γ, β)  →  measure cut
```

`solve_with_prediction` reports three numbers: **baseline** (full optimization),
**predicted** (one forward pass), and **refined** (prediction warm-starting a few
QAOA steps).

---

## 7. Spectral encoding & cross-size transfer — `qklstm/encoding.py`

### 7.1 Why the degree encoding blocks transfer

The original encoding `[deg_0, …, deg_{n-1}, density, |E|]` has length `n + 2`. Two
problems follow. Relabeling vertices by a permutation `σ` permutes the degree block, so
the same graph maps to different vectors — the network must spend capacity learning an
invariance. And the length depends on `n`, so a model trained at one size cannot even be
*evaluated* at another.

### 7.2 The normalized Laplacian

```
L = I − D^{-1/2} A D^{-1/2}
```

Its spectrum satisfies `0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ ≤ 2` for **every** graph, independent of
size. This bound is the crucial property: it makes spectra of a 4-node and a 40-node
graph directly comparable. The combinatorial Laplacian `L = D − A` has eigenvalues that
grow with the degrees and would not be comparable across sizes.

Useful facts used as features:

- the multiplicity of eigenvalue `0` equals the number of connected components;
- `λ₂` (algebraic connectivity) measures how well connected the graph is;
- eigenvalues are invariant under vertex relabeling, since `L → P L Pᵀ` is a similarity
  transform and similar matrices share a spectrum. **This is the source of permutation
  invariance** — no learning required.

Isolated vertices use the convention `D^{-1/2} → 0`.

### 7.3 Fixed-length summary

The spectrum has `n` values, which varies. To obtain a constant-length descriptor we take
**quantiles at fixed positions** `q = 0.0, 0.1, …, 1.0`, giving 11 numbers that summarize
the shape of the spectral density for any `n`. Adding mean, std, `λ₂` and the zero-
eigenvalue fraction, plus degree moments normalized by `n − 1` and the scale-free
structural statistics (density, clustering, triangle density, assortativity), yields a
**24-dimensional** vector (25 with the optional `log(n)/10` size hint) for every graph.

Since every entry is a function of the unordered spectrum or a normalized aggregate over
vertices/edges, the encoding is permutation-invariant by construction.

---

## 8. QAOA parameter symmetries — `qklstm/qaoa.py`

Learning to predict `(γ, β)` only works if the targets are consistent. The MaxCut QAOA
objective `F(γ, β)` is invariant under three transformations:

| symmetry | why |
| --- | --- |
| `γ_l → γ_l + π` | `H_C` has an integer spectrum, so `e^{-iγH_C}` is `π`-periodic up to a global phase |
| `β_l → β_l + π/2` | `e^{-i(π/2)ΣX} ∝ X^{⊗n}`, which commutes with `H_C` and satisfies `X^{⊗n}|+⟩^{⊗n} = |+⟩^{⊗n}` |
| `(γ, β) → (−γ, −β)` | `H_C` is real-diagonal and `|+⟩^{⊗n}` is real, so `F` is invariant under complex conjugation (applies jointly to all layers) |

Consequently the optimal angles form an **orbit** of equivalent solutions. A classical
optimizer started at a random point returns an arbitrary member of that orbit, so labels
gathered across problems have standard deviation ≈ 1.8 rad even when every one of them is
optimal. Minimizing `‖θ̂ − θ*‖²` against such targets drives the prediction toward the
*centroid* of several valid solutions, which is not itself a solution — empirically this
scored *below random*.

**Canonicalization.** `canonicalize_qaoa_params` folds each label onto one representative:

```
γ ← γ mod π
β ← β mod π/2
if β₁ > π/4:  (γ, β) ← (−γ mod π, −β mod π/2)      # joint time reversal
```

giving the fundamental domain `γ ∈ [0, π)`, `β ∈ [0, π/4]`. This changes no objective
value (verified to ~5e-15) and collapses label scatter to exactly zero.

**Periodic loss.** Even canonicalized, angles live on a circle: `γ = 0.01` and
`γ = π − 0.01` are nearly the same circuit but maximally far apart in MSE. The training
objective is therefore

```
L = mean[ 1 − cos(2π·Δγ / π) ] + mean[ 1 − cos(2π·Δβ / (π/2)) ]
  = mean[ 1 − cos(2Δγ) ] + mean[ 1 − cos(4Δβ) ]
```

which vanishes exactly when the prediction matches the target up to a full period, and is
smooth and differentiable everywhere.

---

## 9. Complexity & intuition

- **Classical QAOA per problem:** `O(I · 2ⁿ)` for `I` optimization iterations.
- **QK-LSTM inference:** one forward pass + kernel evals — no per-problem
  optimization loop, giving a large speedup once trained.
- **Why it works:** individual MaxCut landscapes are non-convex, but a *family*
  of problems shares learnable global structure. The LSTM captures the temporal
  "flow" toward good angles; the quantum kernel supplies a rich similarity space;
  together they provide strong warm starts.

```
Performance ≈ α·(quantum nonlinearity) + β·(classical learning) + γ·(interaction)
```

The interaction term — quantum features feeding a learned sequence model — is the
novel ingredient.
