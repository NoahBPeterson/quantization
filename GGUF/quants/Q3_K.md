# GGUF Q3_K Quantization

## Before Quantization: Floating-Point Weights (BF16)

The layer consists of 2048 floating-point weights in BF16 (BFloat16) format:

For example, consider an array of floating-point weights for a layer before quantization:

    [1.25, -0.5, 2.3, -1.1, ..., 1.0, -0.9]  <-- 2048 weights in total

 * Each BF16 weight occupies 16 bits (2 bytes).
 * Total size before quantization: 2048 × 2 = 4096 bytes.

## After Quantization: Q3_K Format

In Q3_K quantization, the weights are processed in super-blocks and blocks using 3-bit quantization (via k‑quantization). Each super-block contains 256 weights that are further divided into 16 blocks of 16 weights each. The quantized weights are represented with 3 bits per weight, where the lower 2 bits are stored in a packed array and the higher bit (if the quantized value exceeds 3) is stored separately in a bitmask. In addition, each block has a local scale (a 6‑bit value) and there is a super-block–wide global scale factor.

The dequantized weight is recovered as:

  w = q * block_scale

where the block scale is derived from a packed 6‑bit value.

### Step 1: Divide into Super-Blocks and Blocks

- **Super-Blocks:**  
  The 2048 weights are divided into 8 super-blocks, each containing 256 weights.

- **Blocks:**  
  Each super-block is subdivided into 16 blocks of 16 weights.

### Step 2: Compute and Pack Block Scales

For each 16-weight block in a super-block:

- Compute a local scale (using, for example, a function like `make_q3_quants`) which yields a scale value in a range such that the absolute values are within [−32, 31].
- Determine the maximum scale in the super-block.  
- Compute a global scaling factor (d) as the inverse of a relative maximum (stored as a 16-bit FP16 value).
- Normalize and quantize each block's local scale to 6 bits (after adding an offset of 32).  
- Pack all 16 local block scales into 12 bytes using the following pattern:

  ```
    Byte 0: IIIIAAAA
    Byte 1: JJJJBBBB
    Byte 2: KKKKCCCC
    Byte 3: LLLLDDDD
    Byte 4: MMMMEEEE
    Byte 5: NNNNFFFF
    Byte 6: OOOOGGGG
    Byte 7: PPPPHHHH
    Byte 8: MMIIEEAA
    Byte 9: NNJJFFBB
    Byte 10: OOKKGGCC
    Byte 11: PPLLHHDD
  ```

  Each letter represents 4 or 2 bits such that the total per block is 6 bits; once unpacked in the dequantization process, these yield 16 signed scale values (after subtracting the offset of 32).

### Step 3: Quantize Each Weight to 3 Bits

For each weight within a 16-weight block:

- Normalize the weight using the corresponding block's scale.
- Compute a quantized value by rounding the normalized result.  
- Clamp the result to the range [-4, 3] and then add 4 to bring the value into [0, 7].

### Step 4: Pack Quantized Weights and Upper Bits

- **Packed Low Bits (qs):**  
  After adjusting, the lower 2 bits (values in the range [0, 3]) of each quantized value are packed. Four such 2‑bit values are stored in one byte. For each super-block, this produces 64 bytes (since 256 weights × 2 bits = 512 bits = 64 bytes).

  For example, given these 16 weights:
  ```
  weights = [
       0, 1, 2, 3,  # These remain unchanged (no upper bits set)
       4, 5, 6, 7,  # Upper bits set; these become: 4->0, 5->1, 6->2, 7->3 (low parts)
       3, 2, 1, 0,  # No upper bits set
       7, 6, 5, 4   # Upper bits set; these become: 7->3, 6->2, 5->1, 4->0 (low parts)
  ]
  ```

  The lower 2 bits are packed four values per byte:
  ```
  qs[0] = (w[3] & 3) << 6 | (w[2] & 3) << 4 | (w[1] & 3) << 2 | (w[0] & 3)
        = (3 << 6) | (2 << 4) | (1 << 2) | 0
        = 0xE4  # 11100100 in binary

  qs[1] = (w[7] & 3) << 6 | (w[6] & 3) << 4 | (w[5] & 3) << 2 | (w[4] & 3)
        = (3 << 6) | (2 << 4) | (1 << 2) | 0
        = 0xE4  # 11100100 in binary

  qs[2] = (w[11] & 3) << 6 | (w[10] & 3) << 4 | (w[9] & 3) << 2 | (w[8] & 3)
        = (0 << 6) | (1 << 4) | (2 << 2) | 3
        = 0x1B  # 00011011 in binary

  qs[3] = (w[15] & 3) << 6 | (w[14] & 3) << 4 | (w[13] & 3) << 2 | (w[12] & 3)
        = (0 << 6) | (1 << 4) | (2 << 2) | 3
        = 0x1B  # 00011011 in binary
  ```

- **Upper Bit Storage (qh):**  
  For each weight, if its quantized value is greater than 3 (i.e. if the third bit is set), store this information in a bit array. Using the same example weights:
  ```
  Upper bits = [
      0,0,0,0,  # First four values (0-3) have no upper bits set
      1,1,1,1,  # Next four values (4-7) have upper bits set
      0,0,0,0,  # Next four values (3,2,1,0) have no upper bits set
      1,1,1,1   # Last four values (7,6,5,4) have upper bits set
  ]
  ```
  
  These bits are packed into a 16-bit value:
  ```
  qh = 0xF0F0  # 1111000011110000 in binary
  ```

  This bit pattern shows exactly which values were originally above 3 and needed their upper bits stored separately.

### Step 5: Store Super-Block Metadata

For each super-block:
- **Block Scales:** The 16 local block scales are packed into 12 bytes.
- **Global Scale (d):** A 16-bit FP16 value (2 bytes) computed from the maximum local scale.
- **Upper Bits (qh):** 32 bytes storing the third bit of each quantized value.
- **Packed Low Bits (qs):** 64 bytes storing pairs of 2-bit values.

Thus, each super-block consumes a total of 12 + 2 + 32 + 64 = 110 bytes.

## Final Memory Layout

Since there are 8 super-blocks in the layer, the quantized representation occupies:

| Section                      | Data Size per Super-Block | Total Size (8 Super-Blocks) |
|------------------------------|---------------------------|-----------------------------|
| Quantized Weights (qs)       | 64 bytes                  | 64 × 8 = 512 bytes          |
| Upper Bits (qh)              | 32 bytes                  | 32 × 8 = 256 bytes          |
| Block Scales (packed)        | 12 bytes                  | 12 × 8 = 96 bytes           |
| Global Scale (d)             | 2 bytes                   | 2 × 8 = 16 bytes            |
| **Total per Super-Block**    | **110 bytes**             | **880 bytes**               |

## Comparison: Before and After Quantization

| Data                                   | Format         | Total Size in Bytes |
|----------------------------------------|----------------|---------------------|
| Original Floating-Point Weights        | BF16 (16 bits) | 4096 bytes          |
| Quantized Weights (Q3_K)               | Q3_K (3.44 bits)          | 880 bytes           |

For Q3_K:

    Layer (2048 weights)
    ├── Super-Block 1 (256 weights, 17 scale values: 16 local block scales + 1 global scale)
    │   ├── Block 1 (16 weights, local scale)
    │   ├── Block 2 (16 weights, local scale)
    │   └── ...
    │   └── Block 16 (16 weights, local scale)
    ├── Super-Block 2 (256 weights, 17 scales)
    ├── ...
    └── Super-Block 8 (256 weights, 17 scales)

## Limitations

- Q3_K quantization requires the total number of weights in a layer to be divisible by 256 (the super-block size).  
- Layers with non-divisible weight counts cannot be quantized using Q3_K without additional padding or trimming.
- The format uses round-to-nearest quantization.
