#!/usr/bin/env python3
"""
DCT-Based Image Compression — Main Entry Point

Demonstrates block-based DCT compression at multiple quality levels,
mirroring the core pipeline used in JPEG and video codecs (H.264, HEVC, VVC).

Usage:
    python main.py                      # use generated test image
    python main.py path/to/image.png    # use custom image
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from dct_compressor import compress_image, calculate_ssim
from visualize import plot_comparison, plot_dct_coefficients, plot_rate_distortion

QUALITY_LEVELS = [10, 30, 50, 70, 90]
SAMPLE_IMAGE_PATH = Path(__file__).parent / "sample_image.png"


def generate_test_image(path: Path, size: int = 256) -> None:
    """Generate a test image with gradients, edges, and textures.

    This gives a meaningful signal for DCT analysis — smooth regions compress
    well, sharp edges and textures reveal quantization artifacts.
    """
    img = np.zeros((size, size), dtype=np.uint8)

    # Horizontal gradient (top-left quadrant)
    for j in range(size // 2):
        img[:size//2, j] = int(j / (size // 2) * 255)

    # Checkerboard pattern (top-right quadrant)
    block = 8
    for i in range(size // 2):
        for j in range(size // 2, size):
            if ((i // block) + ((j - size // 2) // block)) % 2 == 0:
                img[i, j] = 200
            else:
                img[i, j] = 55

    # Circular pattern (bottom-left quadrant)
    cy, cx = size * 3 // 4, size // 4
    for i in range(size // 2, size):
        for j in range(size // 2):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            img[i, j] = int(127 + 127 * np.sin(dist * 0.3))

    # Diagonal stripes (bottom-right quadrant)
    for i in range(size // 2, size):
        for j in range(size // 2, size):
            img[i, j] = int(127 + 127 * np.sin((i + j) * 0.2))

    Image.fromarray(img).save(path)
    print(f"Generated test image: {path} ({size}x{size})")


def main():
    # Determine input image
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"Error: {image_path} not found.")
            sys.exit(1)
    else:
        if not SAMPLE_IMAGE_PATH.exists():
            generate_test_image(SAMPLE_IMAGE_PATH)
        image_path = SAMPLE_IMAGE_PATH

    print(f"\nInput image: {image_path}")
    print(f"Quality levels: {QUALITY_LEVELS}")
    print("=" * 60)

    # Run compression at each quality level
    results = []
    for q in QUALITY_LEVELS:
        res = compress_image(str(image_path), quality=q)
        ssim = calculate_ssim(res['original'], res['reconstructed'])
        res['ssim'] = ssim
        results.append(res)

        print(
            f"  Quality {q:3d}  |  "
            f"PSNR: {res['psnr']:6.2f} dB  |  "
            f"SSIM: {ssim:.4f}  |  "
            f"Zero coeffs: {res['zero_ratio']*100:5.1f}%"
        )

    print("=" * 60)

    # Generate visualizations
    print("\nGenerating plots...")
    original = results[0]['original']

    plot_comparison(original, results)

    # Show DCT coefficients for the lowest quality (most aggressive quantization)
    low_q = results[0]
    plot_dct_coefficients(low_q['sample_dct'], low_q['sample_quantized'], low_q['quality'])

    plot_rate_distortion(results)

    print("\nDone! Check the output/ directory for plots.")


if __name__ == "__main__":
    main()
