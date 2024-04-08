// from chatgpt
static inline uint16_t fp32_to_bf16_bits(uint32_t f) {
  /*
   * Extract the sign of the input number into the high bit of the 16-bit word:
   *
   *      +---+-----+-------------------+
   *      | S | EEEE EEEE | MMM MMMM     |
   *      +---+-----+-------------------+
   * Bits  15  14-7          6-0
   */
  const uint32_t sign = f & UINT32_C(0x80000000);
  /*
   * Extract the exponent and the top 7 bits of the mantissa into the bits 0-14
   * of the 16-bit word:
   *
   *      +---+-----+-------------------+
   *      | 0 | EEEE EEEE | MMM MMMM     |
   *      +---+-----+-------------------+
   * Bits  14  7-0          6-0
   */
  const uint32_t rest = (f >> 16) & UINT32_C(0x7FFF);

  // Combine the sign with the rest of the number
  const uint16_t bfloat16 = (sign >> 16) | rest;

  // Handle rounding by examining the bits that are being truncated
  const uint32_t rounding_mask = UINT32_C(0x00007FFF);
  const uint32_t rounding_bits = f & rounding_mask;
  const uint32_t halfway = UINT32_C(0x00004000);
  if (rounding_bits > halfway || (rounding_bits == halfway && (bfloat16 & 1))) {
    // Round up
    return bfloat16 + 1;
  } else {
    // Truncate
    return bfloat16;
  }
}