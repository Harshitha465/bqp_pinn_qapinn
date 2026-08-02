import torch
import torch.nn as nn
import pennylane as qp

from .pinn import ACTIVATIONS
from .quantum_layers import QuantumLayerJax

SUPPORTED_ENCODINGS = ["angle", "reupload"]
SUPPORTED_ANSATZ = ["basic_entangler", "strongly_entangling"]

def build_qnode(n_qubits, n_layers, encoding="angle", ansatz="basic_entangler"):
    """Returns a single-sample PennyLane QNode using the JAX interface."""
    assert encoding in SUPPORTED_ENCODINGS, f"Unsupported encoding: {encoding}"
    assert ansatz in SUPPORTED_ANSATZ, f"Unsupported ansatz: {ansatz}"
    dev = qp.device("default.qubit", wires=n_qubits)

    def apply_ansatz(weights):
        if ansatz == "basic_entangler":
            qp.BasicEntanglerLayers(weights, wires=range(n_qubits))
        else:
            qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(inputs, weights):
        if encoding == "reupload":
            for layer in range(n_layers):
                qp.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                apply_ansatz(weights[layer:layer + 1])
        else:
            qp.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            apply_ansatz(weights)
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    return circuit

class QAPINN(nn.Module):
    def __init__(self, n_qubits=3, n_qlayers=3, classical_hidden=(25, 25, 25, 25),
                 encoding="angle", ansatz="basic_entangler", activation="tanh"):
        super().__init__()
        self.n_qubits, self.n_qlayers = n_qubits, n_qlayers
        self.encoding, self.ansatz = encoding, ansatz
        circuit = build_qnode(n_qubits, n_qlayers, encoding, ansatz)
        weight_shape = (n_qlayers, n_qubits, 3) if ansatz == "strongly_entangling" else (n_qlayers, n_qubits)
        self.quantum_layer = QuantumLayerJax(circuit, weight_shape)
        self.input_projection = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, n_qubits))
        act = ACTIVATIONS[activation.lower()]
        modules, prev = [], n_qubits
        for h in classical_hidden:
            modules += [nn.Linear(prev, h), act()]
            prev = h
        modules.append(nn.Linear(prev, 3))
        self.postprocessing = nn.Sequential(*modules)

    def to(self, device):
        self.input_projection.to(device)
        self.postprocessing.to(device)
        self.quantum_layer.to(device)
        return self

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        projected = self.input_projection(inputs)
        quantum_out = self.quantum_layer(projected)
        # Fixed: Handle cases where JAX/PennyLane batching might transpose dimensions
        if quantum_out.shape[0] == self.n_qubits and quantum_out.shape[1] == projected.shape[0]:
            quantum_out = quantum_out.T
        return self.postprocessing(quantum_out)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
