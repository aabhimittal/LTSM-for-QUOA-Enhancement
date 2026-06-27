"""Quantum-kernel-enhanced LSTM (``QuantumKernelLSTM``).

A standard LSTM is augmented with a *quantum kernel* preprocessing stage.  For
each time step the input vector is compared against a fixed reference set via the
quantum kernel; the resulting similarity vector is projected back to the input
dimension and fused with the original features through a residual connection::

    x_tilde = x + 0.5 * proj( K_Q(x, X_ref) )

The quantum kernel is treated as *fixed preprocessing*: gradients flow through
the projection layer and the LSTM, but not through the (non-differentiable)
quantum circuit evaluation.

Bug fix vs. the original prototype
----------------------------------
In the prototype the projection layer was ``nn.Linear(input_dim, input_dim)``,
but the quantum-enhanced features have dimension ``m_ref`` (the number of
reference samples), not ``input_dim`` -- so the forward pass crashed whenever
``use_quantum=True``.  Here the projection is created lazily in
:meth:`QuantumKernelLSTM.set_reference_data` as ``nn.Linear(m_ref, input_dim)``
so the residual fusion is dimensionally consistent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .quantum_kernel import QuantumKernel


class QuantumKernelLSTM(nn.Module):
    """LSTM with optional quantum-kernel feature enhancement.

    Parameters
    ----------
    input_dim:
        Dimensionality of each time-step feature vector.
    hidden_dim:
        Hidden size of the LSTM.
    output_dim:
        Size of the final prediction (e.g. ``2 * qaoa_depth`` QAOA parameters).
    n_qubits:
        Number of qubits used by the quantum kernel.
    n_layers:
        Number of stacked LSTM layers.
    use_quantum:
        If ``False`` the model is a plain LSTM (useful as an ablation baseline).
    fusion_weight:
        Weight ``alpha`` of the quantum residual in ``x + alpha * proj(K)``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_qubits: int = 4,
        n_layers: int = 1,
        use_quantum: bool = True,
        fusion_weight: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_quantum = use_quantum
        self.fusion_weight = fusion_weight

        # Quantum kernel for feature enhancement.
        if use_quantum:
            self.quantum_kernel = QuantumKernel(n_qubits=n_qubits, n_layers=2)
        else:
            self.quantum_kernel = None

        # The projection ``nn.Linear(m_ref, input_dim)`` is created lazily once
        # the reference set (and hence ``m_ref``) is known.
        self.quantum_projection = None

        # LSTM core.
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )

        # Dense head with a small bottleneck.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Tanh()

        # Reference data for the kernel (mean-pooled over the sequence).
        self.register_buffer("X_ref", None)

    # ------------------------------------------------------------------ #
    # Reference set / lazy projection
    # ------------------------------------------------------------------ #
    def set_reference_data(self, X_ref: torch.Tensor) -> None:
        """Register the reference set used for kernel comparisons.

        ``X_ref`` is expected with shape ``(m_ref, seq_len, input_dim)`` (a batch
        of trajectories); it is mean-pooled over the sequence dimension to a set
        of ``m_ref`` reference vectors.  The quantum projection layer is then
        (re)built as ``nn.Linear(m_ref, input_dim)``.
        """
        ref = X_ref.mean(dim=1).detach()  # (m_ref, input_dim)
        self.X_ref = ref
        m_ref = ref.shape[0]
        # Build the projection on the same device as the reference data.
        self.quantum_projection = nn.Linear(m_ref, self.input_dim).to(ref.device)

    # ------------------------------------------------------------------ #
    # Quantum enhancement
    # ------------------------------------------------------------------ #
    def quantum_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` to quantum-kernel features projected to ``input_dim``.

        Returns a tensor with the same shape as ``x`` (so it can be added as a
        residual).  If quantum mode is off or no reference set has been provided,
        a zero residual is returned.
        """
        if not self.use_quantum or self.X_ref is None:
            return torch.zeros_like(x)

        batch_size, seq_len, _ = x.shape
        x_np = x.detach().cpu().numpy()
        X_ref_np = self.X_ref.detach().cpu().numpy()
        m_ref = len(X_ref_np)

        # Kernel similarity of every (batch, time) vector against the reference.
        K = np.zeros((batch_size, seq_len, m_ref), dtype=np.float32)
        for t in range(seq_len):
            x_t = x_np[:, t, :]
            for i in range(batch_size):
                for j in range(m_ref):
                    K[i, t, j] = self.quantum_kernel.compute_kernel(
                        x_t[i], X_ref_np[j]
                    )

        K_tensor = torch.from_numpy(K).to(x.device)
        # Project kernel features (m_ref -> input_dim) for residual fusion.
        return self.quantum_projection(K_tensor)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantum fusion -> LSTM -> dense head.

        ``x`` has shape ``(batch, seq_len, input_dim)`` and the output has shape
        ``(batch, output_dim)``.
        """
        if self.use_quantum and self.X_ref is not None:
            x = x + self.fusion_weight * self.quantum_enhance(x)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # final time step

        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
