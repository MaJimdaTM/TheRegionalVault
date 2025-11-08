import psutil
import time
import torch

def log_power():
    process = psutil.Process()
    start_time = time.time()
    start_mem = process.memory_info().rss / 1024**2

    model = torch.nn.Linear(128, 10).cuda()
    x = torch.randn(64, 128).cuda()
    y = torch.zeros(64, dtype=torch.long).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    duration = time.time() - start_time
    mem_delta = process.memory_info().rss / 1024**2 - start_mem
    print(f"Energy proxy: {duration:.2f}s, {mem_delta:.1f} MB RAM delta")

if __name__ == "__main__":
    log_power()
