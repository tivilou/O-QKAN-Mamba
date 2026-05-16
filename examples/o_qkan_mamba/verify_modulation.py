"""Verify ontology modulation: same input, different concepts → different spectra."""
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.qkan.experimental.ontology_modulated_daruan import OntologyModulatedDARUAN


def main():
    os.makedirs("outputs/phase_4", exist_ok=True)
    device = "cpu"
    dim = 16
    reps = 4
    d_ontology = 64

    model = OntologyModulatedDARUAN(
        dim=dim, reps=reps, d_ontology=d_ontology, device=device
    )

    x = torch.randn(1, dim)

    # Concept 1: "healthy" — activate first few dimensions
    c_healthy = torch.zeros(1, d_ontology)
    c_healthy[0, :5] = 1.0

    # Concept 2: "septic" — activate different dimensions
    c_septic = torch.zeros(1, d_ontology)
    c_septic[0, 30:40] = 1.0

    # Concept 3: zero (baseline)
    c_zero = torch.zeros(1, d_ontology)

    with torch.no_grad():
        w_healthy = model.compute_modulated_weights(c_healthy)[0]
        w_septic = model.compute_modulated_weights(c_septic)[0]
        w_zero = model.compute_modulated_weights(c_zero)[0]

        out_healthy = model(x, c_healthy)[0]
        out_septic = model(x, c_septic)[0]
        out_zero = model(x, c_zero)[0]

    print("=" * 60)
    print("Ontology Modulation Verification")
    print("=" * 60)
    print(f"\nModel: dim={dim}, reps={reps}, d_ontology={d_ontology}")
    print(f"w_base: {model.w_base.data.numpy()}")

    print(f"\n--- Modulated weights per layer ---")
    print(f"{'Layer':<8} {'Zero':<12} {'Healthy':<12} {'Septic':<12}")
    for l in range(reps):
        print(f"  L{l:<5} {w_zero[l]:.4f}      {w_healthy[l]:.4f}      {w_septic[l]:.4f}")

    diff_hs = (out_healthy - out_septic).abs().mean().item()
    diff_hz = (out_healthy - out_zero).abs().mean().item()
    diff_sz = (out_septic - out_zero).abs().mean().item()

    print(f"\n--- Output differences ---")
    print(f"  |healthy - septic|: {diff_hs:.6f}")
    print(f"  |healthy - zero|:   {diff_hz:.6f}")
    print(f"  |septic - zero|:    {diff_sz:.6f}")
    assert diff_hs > 1e-4, "FAIL: healthy vs septic too similar"
    print("\n  PASS: Different concepts produce different outputs")

    # Save visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Modulated weights per layer
        layers = np.arange(reps)
        width = 0.25
        axes[0].bar(layers - width, w_zero.numpy(), width, label="Zero (baseline)", color="gray")
        axes[0].bar(layers, w_healthy.numpy(), width, label="Healthy", color="green")
        axes[0].bar(layers + width, w_septic.numpy(), width, label="Septic", color="red")
        axes[0].set_xlabel("Re-uploading Layer")
        axes[0].set_ylabel("Modulated Weight")
        axes[0].set_title("Frequency Modulation by Clinical Context")
        axes[0].legend()
        axes[0].set_xticks(layers)

        # Plot 2: Output activation comparison
        x_axis = np.arange(dim)
        axes[1].plot(x_axis, out_zero.numpy(), "o-", label="Zero", color="gray", alpha=0.7)
        axes[1].plot(x_axis, out_healthy.numpy(), "s-", label="Healthy", color="green", alpha=0.7)
        axes[1].plot(x_axis, out_septic.numpy(), "^-", label="Septic", color="red", alpha=0.7)
        axes[1].set_xlabel("Dimension")
        axes[1].set_ylabel("Activation Value")
        axes[1].set_title("Output Spectrum Under Different Contexts")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("outputs/phase_4/spectrum_comparison.png", dpi=150)
        print(f"\n  Plot saved to outputs/phase_4/spectrum_comparison.png")
        plt.close()
    except ImportError:
        print("\n  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
