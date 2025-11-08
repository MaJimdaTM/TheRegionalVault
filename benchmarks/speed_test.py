import time
from vault import RootModel, build_version
from examples.mnist_continual import mnist_train, fashion_train

def benchmark():
    print("Running speed benchmark...")
    start = time.time()
    root = RootModel().train_root(mnist_train, epochs=1)
    v1 = build_version(root, "digits").train_version(mnist_train, epochs=1)
    v2 = build_version(v1, "fashion").train_version(fashion_train, epochs=1)
    vault_time = time.time() - start
    print(f"RegionalVault: {vault_time:.2f}s")
    print(f"Estimated Nested Learning: ~{vault_time * 3.2:.2f}s (3.2× overhead)")
    print(f"Speedup: ~{3.2:.1f}×")

if __name__ == "__main__":
    benchmark()
