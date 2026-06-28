"""QK-LSTM for QAOA Enhancement.

A hybrid quantum-classical framework that uses a Quantum Kernel-based LSTM
(QK-LSTM) to predict good QAOA parameters for MaxCut problems, turning a slow
per-problem optimisation into a fast single forward-pass prediction.

Public API
----------
- :class:`QuantumKernel`        -- ZZ-feature-map quantum kernel.
- :class:`QuantumKernelLSTM`    -- LSTM with quantum-kernel feature fusion.
- :class:`QAOA`                 -- MaxCut QAOA solver.
- :class:`QKLSTMQAOAOptimizer`  -- end-to-end meta-optimiser.
- :func:`visualize_results`     -- 4-panel results figure.
"""

from .quantum_kernel import QuantumKernel
from .qaoa import QAOA
from .qklstm_model import QuantumKernelLSTM
from .optimizer import QKLSTMQAOAOptimizer

__all__ = [
    "QuantumKernel",
    "QAOA",
    "QuantumKernelLSTM",
    "QKLSTMQAOAOptimizer",
    "visualize_results",
]

__version__ = "0.1.0"


def __getattr__(name):
    # Lazily expose ``visualize_results`` so importing the package does not pull
    # in Matplotlib unless plotting is actually requested.
    if name == "visualize_results":
        from .visualization import visualize_results

        return visualize_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
