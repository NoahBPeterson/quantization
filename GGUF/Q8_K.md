# GGUF Q8_K Quantization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating-point weights in BF16 (BFloat16) format:

Let’s say we have an array of floating-point weights (in BF16 format) of a layer before quantization:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9] <-- 2048 weights in total

* Each BF16 weight is 16 bits (2 bytes).
* Total size before quantization: 2048 × 2 = 4096 bytes.

## After Quantization: Q8_K Format

In Q8_K, weights are split into blocks of 256, quantized to 8-bit integer values, and the shared block-wide scale metadata is stored for each block.

Step 1: Divide into Blocks

* The 2048 weights are divided into 8 blocks, each containing 256 weights.

This gives us:

    Block 1          Block 2          ...    Block 8
    [256 wts]        [256 wts]        ...    [256 wts]

Step 2: Compute Block-Wide Scale (d)

For each block of 256 weights:

* Compute the absolute maximum value (amax) of all weights in the block.

* Compute the scaling factor:
    * `iscale = -127 / amax`
    * `d = 1 / iscale`

The scaling factor (d) is stored as an FP16 value for efficient memory usage.

Step 3: Quantize Each Weight to 8 Bits

For each weight `x` in the block, compute:
* `quantized_weight = round(x * iscale)`
* Clamp the result to the range [−127, 127]

Example:

    Original weights: [1.25, -0.5, 2.3, -1.1, ...]
    Quantized (8-bit) weights: [100, -50, 127, -80, ...]

The 8-bit quantized weights are stored directly as signed integers.

Step 4: Compute Block Sums

* Compute block-level sums (bsums) for 16-weight chunks within each block.
* Each block has 16 chunk sums, stored as 16-bit integers.

Step 5: Store Metadata

Each block’s metadata includes:

* Block-Wide Scale (d):
    * Stored as a 16-bit FP16 value.

* Block Sums (bsums):
    * Each chunk’s sum (for 16 weights) is stored as a 16-bit integer.
    * Total: 16 chunks × 2 bytes = 32 bytes per block.

## Final Memory Layout

### Block Memory Layout
| Section |	Data |	Size per Block
|--------------|------------------------------------|------------|
| Quantized Weights |	8-bit signed integers (256 weights) |	256 bytes |
| Block Sums (bsums) |	16 × 16-bit sums |	32 bytes |
| Block-Wide Scale (d) |	One FP16 value |	2 bytes |
| Total per Block |		| 290 bytes |

### Layer Layout

Since we have 8 blocks, the total quantized representation for the layer is:

| Section |	Data Size in Bytes |
|--------------|------------------------------------|
| Quantized Weights |	256 bytes * 8 blocks = 2048 bytes |
| Block Sums |	32 bytes * 8 blocks = 256 bytes |
| Block-Wide Scales |	2 bytes * 8 blocks = 16 bytes |
| Total  |	2320 bytes |

## Comparison: Before and After Quantization

| Data |	Format |	Total Size in Bytes|
|--------------|------------------------------------|------------|
| Original Floating-Point Weights |	BF16 (16 bits) |	4096 bytes|
| Quantized Weights (Q8_K) |	Q8_K |	2320 bytes


Explanation of Q8_K Compression

* Quantized Weights: Stored as 8-bit signed integers, providing 256 distinct quantization levels.
* Block-Wide Scale (d): A single FP16 value per block ensures weights are scaled appropriately.
* Block Sums (bsums): Optional metadata used for more efficient computation of matrix multiplications or weighted sums, especially in certain hardware implementations.


Q8_K Layer:

    Layer (2048 weights)
    ├── Block 1 (256 weights, block-wide scale)
    ├── Block 2 (256 weights, block-wide scale)
    ├── ...
    └── Block 8 (256 weights, block-wide scale)

## Limitations

Q8_K quantization requires the number of weights in a layer to be divisible by 256 (the block size). Layers with non-divisible weight counts cannot be quantized using Q8_K without padding or trimming the weights.

Additionally, Q8_K is primarily used for quantizing intermediate results or specific layers where higher precision is needed. For layers where lower precision (e.g., 6-bit or 4-bit quantization) is acceptable, other quantization formats (e.g., Q6_K or Q4_K) may be more memory-efficient.