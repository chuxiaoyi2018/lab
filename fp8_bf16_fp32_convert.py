import numpy as np

def bf16_to_fp32(bf16):
    # bf16 is represented by a uint16 array
    bf16 = np.asarray(bf16, dtype=np.uint32)
    # Create an empty uint32 array
    fp32 = np.zeros(bf16.shape, dtype=np.uint32)
    # Copy the sign bit (1 bit)
    fp32 = (bf16 & 0x8000) << 16
    # Copy the exponent (8 bits)
    fp32 |= (bf16 & 0x7F80) << 16
    # Copy the mantissa (7 bits) and add the implicit leading 1
    fp32 |= (bf16 & 0x007F) << 16
    # Convert to float32
    return fp32.view(np.float32)

def fp32_to_fp8(fp32):
    # fp32 is represented by a float32 array
    fp32 = np.asarray(fp32, dtype=np.float32)
    # Create an empty uint8 array
    fp8 = np.zeros(fp32.shape, dtype=np.uint8)
    # Get the sign bit
    sign = (fp32.view(np.uint32) & 0x80000000) >> 24
    # Get the exponent and shift it to fit in 4 bits
    exponent = np.maximum(np.minimum(((fp32.view(np.uint32) & 0x7F800000) >> 23) - 112, 15), -15) & 0x0F
    # Get the mantissa and shift it to fit in 3 bits
    mantissa = (fp32.view(np.uint32) & 0x00700000) >> 20
    # Combine sign, exponent, and mantissa
    fp8 = sign | (exponent << 3) | mantissa
    return fp8

def fp8_to_fp32(fp8):
    # fp8 is represented by a uint8 array
    fp8 = np.asarray(fp8, dtype=np.uint8)
    # Create an empty uint32 array
    fp32 = np.zeros(fp8.shape, dtype=np.uint32)
    # Get the sign bit
    sign = (fp8 & 0x80) << 24
    # Get the exponent and shift it to fit in 8 bits
    exponent = ((fp8 & 0x78) >> 3) + 112
    # Get the mantissa and shift it to fit in 23 bits
    mantissa = (fp8 & 0x07) << 20
    # Combine sign, exponent, and mantissa
    fp32 |= sign | (exponent << 23) | mantissa
    # Convert to float32
    return fp32.view(np.float32)

def bf16_to_fp8(bf16):
    # bf16 is represented by a uint16 array
    bf16 = np.asarray(bf16, dtype=np.uint16)
    # Create an empty uint8 array
    fp8 = np.zeros(bf16.shape, dtype=np.uint8)
    # Extract the sign bit
    sign = (bf16 & 0x8000) >> 8
    # Extract the exponent and adjust the bias from 127 (bf16) to 7 (fp8)
    exponent = ((bf16 & 0x7F80) >> 7) - 120
    # Clamp the exponent to fit in 4 bits
    exponent = np.clip(exponent, -8, 7) & 0x0F
    # Extract the mantissa and shift to fit in 3 bits
    mantissa = (bf16 & 0x007F) >> 4
    # Combine sign, exponent, and mantissa
    fp8 = sign | (exponent << 3) | mantissa
    return fp8

# Example usage
bf16_values = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)  # Example bf16 values
fp32_values = bf16_to_fp32(bf16_values)
fp8_values = fp32_to_fp8(fp32_values)
fp8_values_from_bf16 = bf16_to_fp8(bf16_values)
fp32_converted_back = fp8_to_fp32(fp8_values)
fp32_converted_back_from_bf16 = fp8_to_fp32(fp8_values_from_bf16)

print("BF16 values (as FP32):", fp32_values)
print("FP8 values:", fp8_values)
print("FP8 values from bf16:", fp8_values_from_bf16)
print("FP8 values (converted back to FP32):", fp32_converted_back)
print("FP8 values (converted back to FP32 From BF16):", fp32_converted_back_from_bf16)

