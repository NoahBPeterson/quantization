# GGUF Q5_K Quantization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating-point weights in BF16 (BFloat16) format:

For example, consider an array of floating-point weights for a layer before quantization:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9]  <-- 2048 weights in total

- Each BF16 weight occupies 16 bits (2 bytes).
- Total size before quantization: 2048 × 2 = 4096 bytes.

## After Quantization: Q5_K Format

In Q5_K quantization, the weights are processed using a 5‑bit k‑quantization scheme. Here, each weight is represented by a 5‑bit integer value (in the range [0, 31]) and recovered using per-block scale and minimum offset parameters via:

    w = q * block_scale + block_min

where:
- **q** is the 5‑bit quantized value.
- **block_scale** and **block_min** are per-block parameters normalized and quantized to 6 bits.

### Step 1: Divide into Super-Blocks and Blocks

- **Super-Blocks:**  
  The 2048 weights are divided into 8 super-blocks, each containing 256 weights.

- **Blocks:**  
  Each super-block is further subdivided into 8 blocks of 32 weights each.

### Step 2: Compute Block-wise Scales and Minimums

For each 32‑weight block within a super-block:

- **Local Computation:**  
  A local scale (block_scale) and a local minimum (block_min) are computed based on the distribution of the block's weights. Functions such as `make_qkx2_quants` are used, possibly factoring in additional statistics (e.g. average absolute value) for numerical stability.

- **Normalization:**  
  Across the 8 blocks in a super-block, determine:
  - `max_scale`: the maximum local scale.
  - `max_min`: the maximum local minimum.
  
  Then compute inverse normalization factors:
  - `inv_scale = (max_scale > 0) ? 63.f / max_scale : 0.f`
  - `inv_min   = (max_min > 0)   ? 63.f / max_min   : 0.f`
  
  Multiply each block's scale and minimum by the corresponding inverse factor, round to the nearest integer, and clamp the results to 63 (to fit within 6 bits).

- **Packing:**  
  The 6‑bit quantized scales and minimums for the 8 blocks are then packed into a compact metadata array (typically occupying 12 bytes per super‑block).

### Step 3: Quantize Each Weight

For each weight within a block:

- Compute its quantized value using the derived block parameters:
  
    l = round((x + dm) / d)
  
  where:
   * **x** is the original weight.
   * **d** is the effective block scale.
   * **dm** is the effective block minimum.
  
- Clamp the computed value to the range [0, 31] so that it fits in 5 bits.

### Step 4: Pack Quantized Weights into Bytes

The 5‑bit values are split into two parts:

- **Lower 4 Bits (ql):**  
  The lower 4 bits of each quantized weight are packed into a byte array. Two 4‑bit values are combined per byte.

- **High Bit (qh):**  
  For values where the quantized weight exceeds 15, 16 is subtracted and a flag is set. The high (5th) bit is stored in a separate bit‑mask array.

For example, consider a super-block of 256 weights with values cycling from 0 to 31:

    example_weights = [0,1,2,...,31, 0,1,2,...,31, ...]

These weights are processed in 4 chunks of 64 weights each. For each chunk:

1. **Lower 4 bits (ql):**
   - Values 0-15 keep their original lower 4 bits
   - Values 16-31 have 16 subtracted, leaving their lower 4 bits
   This creates a repeating pattern in each chunk:
   ```
   Chunk 0: 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF
   Chunk 1: 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF
   Chunk 2: 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF
   Chunk 3: 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF
   ```
   Each byte contains two 4-bit values. For example, `0x32` represents values 2 and 3.

2. **High bits (qh):**
   - Values 0-15 have their high bit set to 0
   - Values 16-31 have their high bit set to 1
   This creates a pattern where the first half of each 32-value sequence has 0s and the second half has 1s:
   ```
   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF
   ```
   Each byte represents 8 weights' high bits. `0x00` means 8 weights below 16, `0xFF` means 8 weights above 15.

The packing is done in chunks:
- The 256 quantized weights in a super‑block are processed in 4 iterations, each handling 64 weights (two groups of 32).
- In each iteration, for each of the 32 positions:
  - The adjusted lower 4‑bit parts are combined into one byte and stored in `ql`.
  - The corresponding high‑bit flags are accumulated in `qh`.

### Step 5: Store Super-Block Metadata

In addition to the packed weights, each super‑block stores metadata:
- **Global Scale (d):**  
  Computed as `max_scale / 63.f` and stored as a 16‑bit FP16 value (2 bytes).

- **Global Minimum (dmin):**  
  Computed as `max_min / 63.f` and stored as a 16‑bit FP16 value (2 bytes).

- **Block Metadata:**  
  The packed 6‑bit block scales and block minimums together typically occupy 12 bytes per super‑block.

## Final Memory Layout

For each super‑block (256 weights):

| Section                         | Data Format                  | Size per Super‑Block |
|---------------------------------|------------------------------|----------------------|
| Packed Lower Bits (ql)          | 4‑bit values                 | 128 bytes            |
| Packed High Bits (qh)           | Bit‑mask (accumulated flags) | 32 bytes             |
| Block Scales & Minimums         | 6‑bit each, packed           | 12 bytes             |
| Global Scale (d)                | FP16                         | 2 bytes              |
| Global Minimum (dmin)           | FP16                         | 2 bytes              |
| **Total per Super‑Block**       |                              | **176 bytes**        |

For the entire layer (2048 weights; 8 super‑blocks):

| Section                     | Total Size in Bytes   |
|-----------------------------|-----------------------|
| Original Weights            | 4096 bytes            |
| Quantized Weights (Q5_K)    | 8 × 176 = 1408 bytes  |

## Comparison: Before and After Quantization

| Data                                   | Format          | Total Size in Bytes |
|----------------------------------------|-----------------|---------------------|
| Original Floating-Point Weights        | BF16 (16 bits)  | 4096 bytes          |
| Quantized Weights (Q5_K)               | Q5_K (5.5 bits) | 1408 bytes          |

## Limitations

Q5_K quantization requires that the total number of weights in a layer be divisible by 256 (the super‑block size). Layers with non‑divisible weight counts cannot be quantized without padding or trimming.