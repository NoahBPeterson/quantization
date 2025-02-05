# GGUF Q6_K Quantization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating-point weights in BF16 (BFloat16) format:

Let’s say we have an array of floating-point weights (in BF16 format) of a layer before quantization:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9] <-- 2048 weights in total

 * Each BF16 weight is 16 bits (2 bytes).
 * Total size before quantization: 2048 × 2 = 4096 bytes.

## After Quantization: Q6_K Format

In Q6_K, the weights are split into super-blocks, quantized to 6-bit values, and the shared metadata (scales) is stored per block and super-block.

Step 1: Divide into Super-Blocks and Blocks

* The 2048 weights are divided into 8 super-blocks, each containing 256 weights.
* Each super-block of 256 weights is further divided into 16 blocks of 16 weights each.

This gives us:

8 super-blocks in total, with each super-block containing 16 blocks of 16 weights.

    Block 1          Block 2          ...    Block 16
    [16 wts]         [16 wts]         ...    [16 wts]

    Super-Block 1    Super-Block 2    ...    Super-Block 8
    [256 wts]        [256 wts]        ...    [256 wts]

Step 2: Compute Scale for Each Super-Block and Block

For each super-block and its blocks:

* Super-Block-Wide Scale (d):
    * Compute the maximum absolute value of the block scales (max_scale) across all 16 blocks.
    * Store the FP16 inverse scale to normalize block scales:

        * `inverse_scale = -128 / max_scale`
        * `d = 1 / inverse_scale`

* Block Scales:
    * For each 16-weight block, compute the local scale (block_scale) as the maximum absolute value for the block.
    * Normalize and quantize the block scales to 8 bits (int8).

Step 3: Quantize Each Weight to 6 Bits

* Normalize weights within each block using the local block scale:

    * `x' = x / (d × block_scale)`

* Shift and round to fit in the 6-bit range [−32,31]:

    * `quantized_weight = max(-32, min(31, round(x' × block_scale)))`

* Store the quantized weights as adjusted 6-bit values in the range [0,63] for packing:

    * `adjusted_weight = quantized_weight + 32`

Example:

    Original weights: [1.25, -0.5, 2.3, -1.1, ...]
    Quantized (6-bit) weights: [40, 20, 50, 15, ...]  <-- adjusted to fit [0, 63]


Step 4: Pack Quantized Weights into Bytes

* Each 6-bit quantized weight is packed into two arrays:
    * Low 4 bits (ql): Store the lower nibble of the 6-bit value for weights.
    * High 2 bits (qh): Store the upper two bits of four consecutive weights in one byte.

Example:

 * 6-bit weights: [010110, 001101, 001111, 011111, 100001, 000010, 101011, 110101, ...]

    * Lower 4-bits: [0110, 1101, 1111, 1111, 0001, 0010, 1011, 0101, ...] <- Two per byte
        * Packed like this:
        * `ql = [0110_1101, 1111_1111, 0001_0010, 1011_0101, ...]`
    * Upper 2-bits: [01, 00, 00, 01, 10, 00, 10, 11, ...] <- Four per byte
        * Packed like this:
        * `qh = [01_00_00_01, 10_00_10_11, ...]`

Step 5: Store Metadata

Each super-block’s metadata includes:

 * Super-Block-Wide Scale (d):
    * Stored as a 16-bit FP16 value.

 * Block Scales:
    * Each block’s scale is quantized and stored as 8-bit integer.
Total: 16 scales × 1 byte = 16 bytes per super-block.

## Final Memory Layout

### Block Memory Layout
| Section |	Data |	Size per Block
|--------------|------------------------------------|------------|
| Quantized Weights |	Packed 6-bit values |	12 bytes |
| Block Scale (d) |	Block-wide scale in `int8` |	1 byte |
| Total per Block |		| 13 bytes |

### Super-Block Memory Layout
| Section |	Data |	Size per Super-Block
|--------------|------------------------------------|------------|
| Quantized Weights |	Packed 6-bit values |	192 bytes |
| Block Scales | 	8-bit scales for 16 blocks (packed) |	16 bytes |
| Super-Block-Wide Scale (d) |	Super-Block-wide scale in FP16 |	2 bytes |
| Total per Super-Block |		| 210 bytes |

### Layer Layout
Since we have 8 super-blocks, the total quantized representation for the layer is:

| Section |	Data Size in Bytes |
|--------------|------------------------------------|
| Quantized Weights |	192 bytes * 8 super blocks = 1536 bytes |
| Block Scales |	16 bytes * 8 super blocks = 128 bytes |
| Super Block Scales |	2 bytes * 8 super blocks = 16 bytes |
| Total  |	1680 bytes |


## Comparison: Before and After Quantization
| Data |	Format |	Total Size in Bytes|
|--------------|------------------------------------|------------|
| Original Floating-Point Weights |	BF16 (16 bits) |	4096 bytes|
| Quantized Weights (Q6_K) |	Q6_K (6.56 bits) |	1680 bytes

For Q6_K:

    Layer (2048 weights)
    ├── Super-Block 1 (256 weights, 17 scales)
    │   ├── Block 1 (16 weights, local scale)
    │   ├── Block 2 (16 weights, local scale)
    │   └── ...
    │   └── Block 16 (16 weights, local scale)
    │   └── Super-Block-wide scale
    ├── Super-Block 2 (256 weights, 17 scales)
    │   └── ...
    └── Super-Block 8 (256 weights, 17 scales)

## Limitations

Like all K-quants, Q6_K quantization requires the total number of weights in a layer to be divisible by 256 (the super-block size). Layers with non-divisible weight counts cannot be quantized using Q6_K.
