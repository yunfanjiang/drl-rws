import torch.nn as nn


def make_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    n_layers: int,
    activation: str = "ReLU",
) -> nn.Sequential:
    assert n_layers >= 1, f"invalid n_layers provided {n_layers}"

    layers = [
        nn.Linear(in_size, hidden_size),
    ]
    activation = getattr(nn, activation)
    assert activation is not None, f"invalid activation provided {activation}"
    for i in range(n_layers - 1):
        layers.append(activation())
        layers.append(
            nn.Linear(hidden_size, out_size if i == n_layers - 2 else hidden_size)
        )
    return nn.Sequential(*layers)
