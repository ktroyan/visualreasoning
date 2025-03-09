import torch.nn as nn


__all__ = ['get_mlp_head']


def get_mlp_head(head_network_config, embed_dim, output_dim, activation='relu', num_layers=2):
    layers = []

    # Choose activation function
    activation_fn = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU()
    }.get(activation, None)

    if activation_fn is None:
        raise ValueError(f"Activation function '{activation}' not recognized. Choose from ['relu', 'gelu', 'leaky_relu']")
    
    # Input layer
    layers.append(nn.Linear(embed_dim, head_network_config.hidden_dim))
    layers.append(activation_fn)

    # Hidden layers
    hidden_dim = head_network_config.hidden_dim
    for _ in range(num_layers - 2):
        # hidden_dim = max(hidden_dim // 2, 128)  # ensure that the hidden_dim does not become smaller than 128 as the hidden layer dimension shrinks exponentially
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_fn)

    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)