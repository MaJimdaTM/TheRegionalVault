import torch
import torch.nn as nn

class RootModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.frozen = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(self.backbone(x))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True

    def train_root(self, loader, epochs=5, lr=0.001):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                opt.step()
        self.freeze()
        print("Root model trained and frozen.")
