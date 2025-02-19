# GGUF IQ4_NL Quantization

## Overview

IQ4_NL is a non-linear 4-bit quantization scheme that combines block-wise scaling with a fixed lookup table of 16 pre-determined values (`kvalues_iq4nl`). This format achieves 4.25 bits/weight while maintaining low dequantization overhead through:

1. Super-block organization with global and local scaling
2. Binary search quantization to nearest kvalue
3. Efficient 4-bit packing with scale optimization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating‑point weights in BF16 (BFloat16) format:

For example, an array of floating‑point weights for a layer before quantization might be:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9]  <- 2048 weights in total

- Each BF16 weight occupies 16 bits (2 bytes)
- Total size before quantization: 2048 × 2 = 4096 bytes

## After Quantization: IQ4_NL Format

### Step 1: Organize into Super-Blocks
- Divide weights into super-blocks of 256 weights
- Each super-block contains 8 blocks of 32 weights

### Step 2: Compute Block Scales
For each 32-weight block:
1. Find weight with maximum absolute value
2. Calculate initial scale:  
   `d = max_abs / 127` (stored as FP16)
3. Perform iterative optimization (7 tries) to minimize quantization error:
   * For itry in [-7, 7], calculate `id = (itry + kvalues[0])/max_abs;` and use the value of itry that minimizes quantization error.
4. Store optimized scale as 6-bit value (scale_l + scale_h)

### Step 3: Quantize to 4-Bit Indices
For each weight in block:
1. Scale weight: `scaled = weight / d`
2. Binary search to find nearest kvalue:

    kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

3. Store 4-bit index (0-15) of closest match

### Step 4: Pack Data
- Two 4-bit indices packed per byte (128 bytes/super-block)
- Scales stored as:
  - Global scale (d): FP16 (2 bytes)
  - Local scales: 6 bits/block (6 bytes total for 8 blocks)

## Dequantization Process

Weights are reconstructed using:

    w = d * (ls - 32) * kvalues[quantized_index]

Where:
- `d`: FP16 global scale
- `ls`: Local scale reconstructed from 6-bit packed value
- `kvalues`: Predefined lookup table

For each super‑block (256 weights):


## Final Memory Layout

| Section                | Data Format             | Size per Super-Block |
|------------------------|-------------------------|----------------------|
| Global Scale (d)       | FP16                    | 2 bytes              |
| Local Scales           | Packed 6-bit (8 blocks) | 6 bytes              |
| Quantized Weights      | Packed 4-bit (256 wts)  | 128 bytes            |
| **Total per Super-Block** |                      | **136 bytes**        |

## Comparison: Before and After Quantization

| Data                                   | Format             | Total Size in Bytes |
|----------------------------------------|--------------------|---------------------|
| Original Floating‑Point Weights       | BF16 (16 bits)     | 4096 bytes          |
| Quantized Weights (IQ4_NL)            | IQ4_NL (4.25 bits) | 1088 bytes          |

## Limitations
- Requires weight count divisible by 256
- Fixed kvalues limit representation flexibility vs learned codebooks
- Scale reconstruction adds slight computational overhead