# GGUF Q4_K Quantization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating-point weights in BF16 (BFloat16) format:

Let's say we have an array of floating-point weights (in BF16 format) of a layer before quantization:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9]   <-- 2048 weights in total

 * Each BF16 weight is 16 bits (2 bytes).
 * Total size before quantization: 2048 × 2 = 4096 bytes.

## After Quantization: Q4_K Format

In Q4_K, the weights are split into super-blocks, quantized to 4-bit values, and the shared metadata is stored per block and super-block.

Step 1: Divide into Super-Blocks and Blocks

* The 2048 weights are divided into 8 super-blocks, each containing 256 weights.
* Each super-block of 256 weights is further divided into 8 blocks of 32 weights each.

This gives us:

8 super-blocks in total, with each super-block containing 8 blocks of 32 weights.

    Block 1          Block 2          ...    Block 8
    [32 wts]         [32 wts]         ...    [32 wts]

    Super-Block 1    Super-Block 2    ...    Super-Block 8
    [256 wts]        [256 wts]        ...    [256 wts]

Step 2: Compute Scale and Offset for Each Super-Block

Compute Super-Block-Wide Scale (d) and Offset (dmin):

* Calculate a super-block-wide d (scale) and dmin (minimum offset) as 16-bit floats (FP16), representing the maximum scale and offset within that super-block.

Compute 4-Bit Scales and Minimums for Blocks:

* Each 32-weight block has its own local scale and minimum value.
* The scale and minimum for each block are then normalized to fit within 6 bits each.
* These are stored in a compact format where all 8 pairs of 6-bit values are packed together within 12 bytes, resulting in 12 bytes for scales and minimums per super-block.

For example:

| Block | Scale | Offset (Min) |
|---------|-------------|--------------|
| Block 1 | 0.2 | -1.0 |
| Block 2 | 0.15 | -0.8 |
| ... | ... | ... |
| Block 8 | 0.3 | -0.5 |

Step 3: Quantize Each Weight to 4 Bits

Each weight in a block is quantized to a 4-bit integer, fitting values from 0 to 15, using the formula:

    quantized_weight=int((original_weight − block_minimum_value) / block_scale)

Example for Block 1:

    Original weights: [1.25, -0.5, 2.3, -1.1, ...] 
    Quantized (4-bit) weights: [10, 3, 15, 0, ...]  <-- each fits into 4 bits

Step 4: Pack Quantized Weights into Bytes

Since each quantized weight is 4 bits, two weights fit into one byte.

Example:

    4-bit values in Block 1: [10, 3, 15, 0, ...]
    Packed into bytes: [0xA3, 0xF0, ...]

Step 5: Store the Super-Block’s Scale and Offset Metadata

Each super-block’s metadata includes:

 * Super-Block-Wide Scale (d) and Offset (dmin):
    * Both are stored as 16-bit FP16 values, totaling 4 bytes.

 * Normalized Block Scales and Offsets:
    * 8 blocks, each with a 6-bit scale and a 6-bit minimum.
    * Total size for scales and minimums: 12 bytes per super-block.

Example:

    d = 0.2 (scaled down to fit FP16)
    dmin = -1.0 (scaled down to fit FP16)
    scales = [10, 12, ..., 15]  <-- scaled versions of each block’s local scale and local minimum/offset

## Final Memory Layout

| Section |	Data |	Size per Super-Block
|--------------|------------------------------------|------------|
| Quantized Weights |	Packed 4-bit values |	128 bytes |
| Block Scales and Offsets | 	6-bit scales for 8 blocks (packed) |	12 bytes |
| Super-Block-Wide Scale (d) |	Super-Block-wide scale in FP16 |	2 bytes |
| Super-Block-Wide Offset (dmin) |	Super-Block-wide minimum in FP16 |	2 bytes |
| Total per Super-Block |		| 144 bytes |

## Summary for the Entire Layer

Since we have 8 super-blocks, the total quantized representation for the layer is:

| Section |	Data Size in Bytes |
|--------------|------------------------------------|
| Quantized Weights |	128×8=1024 bytes |
| Block Scales and Offset |	12×8=96 bytes |
| Super-Block-Wide Scale and Offset | 	4×8=32 bytes |
| Total  |	1152 bytes |

## Comparison: Before and After Quantization

| Data |	Format |	Total Size in Bytes|
|--------------|------------------------------------|------------|
| Original Floating-Point Weights |	BF16 (16 bits) |	4096 bytes|
| Quantized Weights (Q4_K) |	Q4_K (4.5 bits) |	1152 bytes|

For Q4_K: 

    Layer (2048 weights)
    ├── Super-Block 1 (256 weights, 9 scales, 9 offsets)
    │   ├── Block 1 (32 weights, local scale, and offset)
    │   ├── Block 2 (32 weights, local scale, and offset)
    │   └── ...
    │   └── Block 8 (32 weights, local scale, and offset)
    │   └── Super-Block-wide scale and offset
    ├── Super-Block 2 (256 weights, 9 scales, 9 offsets)
    │   └── ...
    └── Super-Block 16 (256 weights, 9 scales, 9 offsets)

## Limitations: 

Like all K-quants, Q4_K quantization requires the total number of weights in a layer to be divisible by 256 (the block size). Layers with non-divisible weight counts cannot be quantized using Q4_K.