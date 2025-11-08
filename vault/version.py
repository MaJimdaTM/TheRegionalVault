import torch
import torch.nn as nn

class Version(nn.Module):
    def __init__(self, parent, task_name, new_classes=10):
        super().__init__()
        self.parent = parent
        self.task_name = task_name
        self.adapter = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, new_classes)

    def forward(self, x):
        with torch.no_grad():
            parent_feat = self.parent.backbone(x.view(x.size(0), -1))
        adapted = self.adapter(parent_feat)
        return self.head(adapted)

    def train_version(self, loader, epochs=3, lr=0.002):
        self.train()
        self.parent.eval()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                loss = criterion(self(x), y)
                loss.backward()
                opt.step()
        print(f"Version '{self.task_name}' trained.")
