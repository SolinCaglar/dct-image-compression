"""
Visualization module for DCT compression results.

Generates comparison plots, DCT coefficient heatmaps, and rate-distortion curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def plot_comparison(original: np.ndarray, results: list[dict]) -> None:
    """Side-by-side comparison of original vs compressed at different quality levels.

    Args:
        original: Original grayscale image as 2D numpy array.
        results: List of dicts from compress_image(), one per quality level.
    """
    ensure_output_dir()
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))

    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for idx, res in enumerate(results):
        ax = axes[idx + 1]
        ax.imshow(res['reconstructed'], cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f"Quality {res['quality']}\n"
            f"PSNR: {res['psnr']:.1f} dB\n"
            f"Zeros: {res['zero_ratio']*100:.0f}%",
            fontsize=10,
        )
        ax.axis('off')

    fig.suptitle("DCT Compression — Quality Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'comparison.png'}")


def plot_dct_coefficients(sample_dct: np.ndarray, sample_quantized: np.ndarray, quality: int) -> None:
    """Heatmap of DCT coefficients before and after quantization for a single 8x8 block.

    Args:
        sample_dct: Raw DCT coefficients (8x8).
        sample_quantized: Quantized DCT coefficients (8x8).
        quality: Quality factor used for quantization.
    """
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    vmax = max(np.abs(sample_dct).max(), 1)
    im1 = ax1.imshow(sample_dct, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_title("DCT Coefficients\n(before quantization)", fontsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Annotate values
    for i in range(8):
        for j in range(8):
            ax1.text(j, i, f"{sample_dct[i, j]:.0f}", ha='center', va='center', fontsize=6)

    vmax_q = max(np.abs(sample_quantized).max(), 1)
    im2 = ax2.imshow(sample_quantized, cmap='RdBu_r', vmin=-vmax_q, vmax=vmax_q)
    ax2.set_title(f"After Quantization\n(quality={quality})", fontsize=11)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    for i in range(8):
        for j in range(8):
            ax2.text(j, i, f"{sample_quantized[i, j]:.0f}", ha='center', va='center', fontsize=6)

    fig.suptitle("8x8 Block — DCT Coefficient Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dct_coefficients.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'dct_coefficients.png'}")


def plot_rate_distortion(results: list[dict]) -> None:
    """Rate-distortion style curve: quality factor vs PSNR and zero ratio.

    Args:
        results: List of dicts from compress_image().
    """
    ensure_output_dir()
    qualities = [r['quality'] for r in results]
    psnrs = [r['psnr'] for r in results]
    zero_ratios = [r['zero_ratio'] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_psnr = '#2563eb'
    color_zeros = '#dc2626'

    ax1.set_xlabel("Quality Factor", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12, color=color_psnr)
    line1, = ax1.plot(qualities, psnrs, 'o-', color=color_psnr, linewidth=2, markersize=7, label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color_psnr)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Zero Coefficients (%)", fontsize=12, color=color_zeros)
    line2, = ax2.plot(qualities, zero_ratios, 's--', color=color_zeros, linewidth=2, markersize=7, label='Zeros %')
    ax2.tick_params(axis='y', labelcolor=color_zeros)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)

    ax1.set_title("Rate-Distortion Analysis\n(Quality vs PSNR & Compression)", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rate_distortion.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'rate_distortion.png'}")
