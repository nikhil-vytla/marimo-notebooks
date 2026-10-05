import torch.nn as nn


class SelfModelMLP(nn.Module):
    def __init__(self, hidden=512, self_model=False, in_dim=784, n_classes=10):
        super().__init__()
        self.self_model = self_model
        self.n_classes = n_classes
        self.hidden = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, n_classes + (hidden if self_model else 0))

    def forward(self, x):
        h = nn.functional.relu(self.hidden(x))
        out = self.out(h)
        logits = out[:, : self.n_classes]
        a_hat = out[:, self.n_classes :] if self.self_model else None
        return logits, a_hat, h

    def classifier_weight(self):
        return self.out.weight[: self.n_classes]
