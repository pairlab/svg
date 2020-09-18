import torch
import torch.nn as nn

class fc(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers):
        super(fc, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        modules = []
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(nn.ReLU())

        for i in range(n_hidden_layers):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(hidden_size, output_size))

        self.output = nn.Sequential(*modules)

    def forward(self, input):
        return self.output(input)
