import torch
import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU}

class ClassicalPINN(nn.Module):
    """Baseline: FC 2 -> 25 -> 25 -> 25 -> 25 -> 3."""
    def __init__(self, layers=(2, 25, 25, 25, 25, 3), activation="tanh", initializer="xavier"):
        super().__init__()
        self.layers = layers
        act = ACTIVATIONS[activation.lower()]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(act())
        self.net = nn.Sequential(*modules)
        self._init_weights(initializer)

    def _init_weights(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif initializer == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
