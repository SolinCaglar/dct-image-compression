"""
DCT-Based Image Compression Engine

Implements the core pipeline used in JPEG and video codecs (H.264, HEVC, VVC):
  Image → 8x8 blocks → DCT → Quantization → Dequantization → IDCT → Reconstructed Image
"""

import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image

# JPEG standard luminance quantization matrix (ITU-T T.81)
JPEG_QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)


def get_quantization_matrix(quality: int) -> np.ndarray:
    """Scale the JPEG quantization matrix by a quality factor.

    Uses the same scaling formula as libjpeg:
      - quality 50 → use the standard matrix as-is
      - quality < 50 → coarser quantization (more compression)
      - quality > 50 → finer quantization (less compression)

    Args:
        quality: Integer 1-99.  Higher = better quality / less compression.
    """
    quality = np.clip(quality, 1, 99)
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    qm = np.floor((JPEG_QUANTIZATION_MATRIX * scale + 50) / 100)
    qm[qm < 1] = 1
    return qm


def apply_dct_2d(block: np.ndarray) -> np.ndarray:
    """Apply 2D Type-II DCT to an 8x8 block (orthonormalized)."""
    return dctn(block, type=2, norm='ortho')


def apply_idct_2d(block: np.ndarray) -> np.ndarray:
    """Apply inverse 2D DCT (Type-III) to reconstruct an 8x8 block."""
    return idctn(block, type=2, norm='ortho')


def quantize(dct_block: np.ndarray, quality: int) -> np.ndarray:
    """Quantize DCT coefficients by dividing by the scaled quantization matrix and rounding."""
    qm = get_quantization_matrix(quality)
    return np.round(dct_block / qm)


def dequantize(quantized_block: np.ndarray, quality: int) -> np.ndarray:
    """Dequantize by multiplying back with the scaled quantization matrix."""
    qm = get_quantization_matrix(quality)
    return quantized_block * qm


def compress_image(image_path: str, quality: int) -> dict:
    """Full DCT compression pipeline.

    Args:
        image_path: Path to the input image.
        quality: 1-99 quality factor (higher = better quality).

    Returns:
        Dictionary with original image, reconstructed image, and per-block stats.
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    original = np.array(img, dtype=np.float64)

    h, w = original.shape

    # Pad to multiples of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(original, ((0, pad_h), (0, pad_w)), mode='edge')

    ph, pw = padded.shape
    reconstructed = np.zeros_like(padded)

    total_coeffs = 0
    total_zeros = 0
    sample_dct = None
    sample_quantized = None

    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = padded[i:i+8, j:j+8]

            # Level shift: center around zero (standard JPEG step)
            block_shifted = block - 128.0

            # Forward DCT
            dct_block = apply_dct_2d(block_shifted)

            # Quantize
            q_block = quantize(dct_block, quality)

            # Save first block for visualization
            if sample_dct is None:
                sample_dct = dct_block.copy()
                sample_quantized = q_block.copy()

            # Stats
            total_coeffs += 64
            total_zeros += np.count_nonzero(q_block == 0)

            # Dequantize
            dq_block = dequantize(q_block, quality)

            # Inverse DCT
            idct_block = apply_idct_2d(dq_block)

            # Reverse level shift and clip
            reconstructed[i:i+8, j:j+8] = idct_block + 128.0

    # Remove padding and clip to valid pixel range
    reconstructed = np.clip(reconstructed[:h, :w], 0, 255)

    psnr = calculate_psnr(original, reconstructed)
    zero_ratio = total_zeros / total_coeffs

    return {
        'original': original,
        'reconstructed': reconstructed,
        'psnr': psnr,
        'zero_ratio': zero_ratio,
        'total_coeffs': total_coeffs,
        'total_zeros': total_zeros,
        'sample_dct': sample_dct,
        'sample_quantized': sample_quantized,
        'quality': quality,
    }


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between original and compressed images.

    PSNR = 10 * log10(MAX^2 / MSE)
    Standard quality metric used across video coding (H.264, HEVC, VVC).
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute Structural Similarity Index (simplified, per-image).

    Uses the standard SSIM formula with default constants:
      C1 = (0.01 * 255)^2,  C2 = (0.03 * 255)^2
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = np.mean(original)
    mu_y = np.mean(compressed)
    sigma_x_sq = np.var(original)
    sigma_y_sq = np.var(compressed)
    sigma_xy = np.mean((original - mu_x) * (compressed - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return numerator / denominator
