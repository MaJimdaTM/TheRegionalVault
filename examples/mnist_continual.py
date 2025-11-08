import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vault import RootModel, build_version

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
mnist_test = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=64)
fashion_train = DataLoader(datasets.FashionMNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

print("Training Root on MNIST...")
root = RootModel(input_dim=784, output_dim=10)
root.train_root(mnist_train, epochs=2)

print("Adding Version: Digits")
v1 = build_version(root, "digits", new_classes=10)
v1.train_version(mnist_test, epochs=1)

print("Adding Version: Fashion")
v2 = build_version(v1, "fashion", new_classes=10)
v2.train_version(fashion_train, epochs=1)

# Test no forgetting
acc = 0
with torch.no_grad():
    for x, y in mnist_test:
        pred = v2(x).argmax(1)
        acc += (pred == y).sum().item()
print(f"Accuracy on original MNIST after 2 tasks: {acc/len(mnist_test.dataset):.1%}")
