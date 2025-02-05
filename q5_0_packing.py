#!/usr/bin/env python3
def pack_q5_0(weights):
    """
    Pack 32 5-bit weights into a compact format for Q5_0 quantization.
    Each weight must be in range [0, 31] (5 bits).
    Returns:
        - qs: array of 16 bytes containing packed lower 4 bits
        - qh: 32-bit integer containing all high bits
    """
    if len(weights) != 32:
        raise ValueError("Must provide exactly 32 weights")
    if not all(0 <= w <= 31 for w in weights):
        raise ValueError("All weights must be in range [0, 31]")

    # Initialize output arrays
    qs = [0] * 16  # Will hold 16 bytes for packed lower bits
    qh = 0         # Will hold 32 high bits

    # Step 2: Pack the lower 4 bits into bytes
    for j in range(16):
        # Extract lower 4 bits from first and second half
        lb0 = weights[j] & 0x0F
        lb1 = weights[j + 16] & 0x0F
        
        # Combine into one byte: (lb1 << 4) | lb0
        packed_byte = (lb1 << 4) | lb0
        qs[j] = packed_byte

    # Step 3: Pack the high bits into a 32-bit integer
    for j in range(16):
        # Extract and place high bit from first half
        hb0 = (weights[j] & 0x10) >> 4
        qh |= (hb0 << j)

        # Extract and place high bit from second half
        hb1 = (weights[j + 16] & 0x10) >> 4
        qh |= (hb1 << (j + 16))

    return qs, qh

def main():
    # Example weights from the documentation
    weights = [0] * 32  # Initialize all to 0
    
    # Set specific example values
    weights[0] = 17
    weights[1] = 6
    weights[2] = 31
    weights[3] = 2
    weights[4] = 5
    weights[5] = 3
    weights[6] = 0
    weights[7] = 30
    weights[8] = 15
    weights[9] = 14
    weights[10] = 13
    weights[11] = 12
    weights[12] = 11
    weights[13] = 10
    weights[14] = 9
    weights[15] = 8
    weights[16] = 7
    weights[17] = 6
    weights[18] = 18
    weights[19] = 17
    weights[20] = 16
    weights[21] = 15
    weights[22] = 14
    weights[23] = 13
    weights[24] = 12
    weights[25] = 30
    weights[26] = 29
    weights[27] = 28
    weights[28] = 27
    weights[29] = 26
    weights[30] = 25
    weights[31] = 24
    
    # Pack the weights
    qs, qh = pack_q5_0(weights)
    
    # Print results in hexadecimal
    print("Packed lower 4 bits (qs) in hex:")
    for i, byte in enumerate(qs):
        print(f"qs[{i:2d}] = 0x{byte:02X}")
    
    print("\nPacked high bits (qh) in hex:")
    print(f"qh = 0x{qh:08X}")

if __name__ == "__main__":
    main() 