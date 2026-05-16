"""Train TinyQMambaLM on Tiny Shakespeare for Phase 1 validation."""
import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.tiny_qmamba_lm import TinyQMambaLM


def download_tiny_shakespeare(data_dir: str) -> str:
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_data(path: str, block_size: int, device: str):
    with open(path, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, chars


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
    path = download_tiny_shakespeare(data_dir)
    train_data, val_data, vocab_size, chars = load_data(path, args.block_size, device)
    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data)}")

    use_mamba = not args.baseline and device == "cuda"
    model = TinyQMambaLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        latent_dim=args.latent_dim,
        reps=args.reps,
        use_mamba=use_mamba,
        solver="exact",
        device=device,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")
    print(f"Using Mamba: {model.blocks[0].use_mamba}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    times = []

    model.train()
    for step in range(args.steps):
        t0 = time.time()
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dt = time.time() - t0
        times.append(dt)
        losses.append(loss.item())

        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | dt {dt*1000:.1f}ms")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

    out_dir = os.path.join(os.path.dirname(__file__), "../../outputs/phase_1")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "param_count": param_count,
        "avg_step_time_ms": sum(times) / len(times) * 1000,
        "peak_memory_mb": peak_mem,
        "use_mamba": model.blocks[0].use_mamba,
        "device": device,
        "steps": args.steps,
        "converged": losses[-1] < losses[0],
    }

    import json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "losses.txt"), "w") as f:
        for i, l in enumerate(losses):
            f.write(f"{i},{l:.6f}\n")

    print(f"\nResults saved to {out_dir}")
    print(f"Final loss: {losses[-1]:.4f} (initial: {losses[0]:.4f})")
    print(f"Avg step time: {results['avg_step_time_ms']:.1f}ms")
    print(f"Peak GPU memory: {peak_mem:.0f}MB")


if __name__ == "__main__":
    main()
