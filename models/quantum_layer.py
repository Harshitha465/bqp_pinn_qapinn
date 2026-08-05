import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

try:
    JAX_GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except Exception:
    JAX_GPU_AVAILABLE = False


def _torch_to_jax(t):
    t = t.contiguous()
    if t.device.type == "cuda" and JAX_GPU_AVAILABLE:
        try:
            return jax.dlpack.from_dlpack(torch.to_dlpack(t))
        except Exception:
            pass
    return jnp.asarray(t.detach().cpu().numpy())


def _jax_to_torch(a, like):
    if like.device.type == "cuda" and JAX_GPU_AVAILABLE:
        try:
            out = torch.from_dlpack(jax.dlpack.to_dlpack(a))
            return out.to(dtype=like.dtype, device=like.device)
        except Exception:
            pass
    return torch.as_tensor(np.asarray(a), dtype=like.dtype, device=like.device)


class _QuantumJaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, circuit_fn):
        inputs_d, weights_d = inputs.detach(), weights.detach()
        inputs_jax = _torch_to_jax(inputs_d)
        weights_jax = _torch_to_jax(weights_d)
        out_jax, vjp_fn = jax.vjp(circuit_fn, inputs_jax, weights_jax)
        ctx.vjp_fn = vjp_fn
        ctx.inputs_like = inputs_d
        ctx.weights_like = weights_d
        return _jax_to_torch(out_jax, inputs_d)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_jax = _torch_to_jax(grad_output.detach())
        grad_inputs_jax, grad_weights_jax = ctx.vjp_fn(grad_output_jax)
        grad_inputs = _jax_to_torch(grad_inputs_jax, ctx.inputs_like)
        grad_weights = _jax_to_torch(grad_weights_jax, ctx.weights_like)
        return grad_inputs, grad_weights, None


class QuantumLayerJax(nn.Module):
    """
    PennyLane + JAX differentiable quantum layer for PyTorch integration.
    Wraps JAX-jitted batched quantum circuit calls with PyTorch autograd via DLPack / NumPy.
    """
    def __init__(self, single_circuit, weight_shape):
        super().__init__()

        def stacked_circuit(inputs, weights):
            return jnp.stack(single_circuit(inputs, weights))

        self.batched_circuit = jax.jit(jax.vmap(stacked_circuit, in_axes=(0, None)))
        self.weights = nn.Parameter(torch.empty(weight_shape).uniform_(-np.pi, np.pi))

    def forward(self, inputs):
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)
        out = _QuantumJaxFunction.apply(inputs, self.weights, self.batched_circuit)
        return out.squeeze(0) if squeeze else out
