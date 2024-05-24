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



typedef union {
  float    fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp  : 8;  // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

static inline uint32_t bf16_to_fp32_bits(uint16_t bf16) {
    uint32_t result;

    // 保留符号位并将其移到 float32 的符号位位置
    result = (uint32_t)(bf16 & 0x8000) << 16;

    // 复制指数位并将它们移到 float32 的指数位位置
    // BFloat16 的指数位与 float32 的指数位是对齐的，所以直接复制即可
    result |= (uint32_t)(bf16 & 0x7F80) << 16;

    // 将尾数位移到 float32 的尾数位位置
    // BFloat16 的尾数位需要左移（16 - 7 = 9）位
    result |= (uint32_t)(bf16 & 0x007F) << 16;

    return result;
}

static inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
	/*
	 * Extend the half-precision floating-point number to 32 bits and shift to the upper part of the 32-bit word:
	 *      +---+-----+------------+-------------------+
	 *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
	 *      +---+-----+------------+-------------------+
	 * Bits  31  26-30    16-25            0-15
	 *
	 * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
	 */
	const uint32_t w = (uint32_t) h << 16;
	/*
	 * Extract the sign of the input number into the high bit of the 32-bit word:
	 *
	 *      +---+----------------------------------+
	 *      | S |0000000 00000000 00000000 00000000|
	 *      +---+----------------------------------+
	 * Bits  31                 0-31
	 */
	const uint32_t sign = w & UINT32_C(0x80000000);
	/*
	 * Extract mantissa and biased exponent of the input number into the bits 0-30 of the 32-bit word:
	 *
	 *      +---+-----+------------+-------------------+
	 *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
	 *      +---+-----+------------+-------------------+
	 * Bits  30  27-31     17-26            0-16
	 */
	const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
	/*
	 * Renorm shift is the number of bits to shift mantissa left to make the half-precision number normalized.
	 * If the initial number is normalized, some of its high 6 bits (sign == 0 and 5-bit exponent) equals one.
	 * In this case renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note that if we shift
	 * denormalized nonsign by renorm_shift, the unit bit of mantissa will shift into exponent, turning the
	 * biased exponent into 1, and making mantissa normalized (i.e. without leading 1).
	 */
#ifdef _MSC_VER
	unsigned long nonsign_bsr;
	_BitScanReverse(&nonsign_bsr, (unsigned long) nonsign);
	uint32_t renorm_shift = (uint32_t) nonsign_bsr ^ 31;
#else
	uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
	renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
	/*
	 * Iff half-precision number has exponent of 15, the addition overflows it into bit 31,
	 * and the subsequent shift turns the high 9 bits into 1. Thus
	 *   inf_nan_mask ==
	 *                   0x7F800000 if the half-precision number had exponent of 15 (i.e. was NaN or infinity)
	 *                   0x00000000 otherwise
	 */
	const int32_t inf_nan_mask = ((int32_t) (nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
	/*
	 * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31 into 1. Otherwise, bit 31 remains 0.
	 * The signed shift right by 31 broadcasts bit 31 into all bits of the zero_mask. Thus
	 *   zero_mask ==
	 *                0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
	 *                0x00000000 otherwise
	 */
	const int32_t zero_mask = (int32_t) (nonsign - 1) >> 31;
	/*
	 * 1. Shift nonsign left by renorm_shift to normalize it (if the input was denormal)
	 * 2. Shift nonsign right by 3 so the exponent (5 bits originally) becomes an 8-bit field and 10-bit mantissa
	 *    shifts into the 10 high bits of the 23-bit mantissa of IEEE single-precision number.
	 * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the different in exponent bias
	 *    (0x7F for single-precision number less 0xF for half-precision number).
	 * 4. Subtract renorm_shift from the exponent (starting at bit 23) to account for renormalization. As renorm_shift
	 *    is less than 0x70, this can be combined with step 3.
	 * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the input was NaN or infinity.
	 * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent into zero if the input was zero. 
	 * 7. Combine with the sign of the input number.
	 */
	return sign | ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) | inf_nan_mask) & ~zero_mask);
}

void dump_bf16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset) {
  fp32 test;
  test.bits = bf16_to_fp32_bits(ATTENTION_MASK);
  std::cout << test.fval << std::endl;
  std::cout << "-------------------------------------" << std::endl;

  int size = 10;
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s(bm_handle, data.data(), mem);
  std::cout << "-------------------------------------" << std::endl;
  fp32 t;
  // 使用最后一个数据作为示例
  t.bits = bf16_to_fp32_bits(data[0]);
  std::cout << t.fval << std::endl;
  // 打印前 10 个数据
  for(int i = 0; i < 10; i++){
    t.bits = bf16_to_fp32_bits(data[i]);
    std::cout << t.fval << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;
}

void dump_fp16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset) {
  int size = 10;
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s_partial_offset(bm_handle, data.data(),mem, size, offset);
  std::cout << "-------------------------------------" << std::endl;
  fp32 t;
  // 使用最后一个数据作为示例
  t.bits = fp16_ieee_to_fp32_bits(data[0]);
  std::cout << t.fval << std::endl;
  // 打印前 10 个数据
  for(int i = 0; i < 10; i++){
    t.bits = fp16_ieee_to_fp32_bits(data[i]);
    std::cout << t.fval << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;
}


void dump_fp32_tensor(bm_handle_t bm_handle, bm_device_mem_t &mem, int offset) {
  int size = 10;
  std::vector<float> data(size);
  bm_memcpy_d2s(bm_handle, data.data(), mem);
  std::cout<<"-------------------------------------"<<std::endl;
  std::cout<< data[data.size()-1] << std::endl;
  for(int i=0;i<10;i++){
    std::cout<< data[i] << std::endl;
  }
  std::cout<<"-------------------------------------"<<std::endl;
  auto ptr = data.data();
  ptr[0] = ptr[0];
}

void dump_int_tensor(bm_handle_t bm_handle, bm_device_mem_t &mem, int offset) {
  int size = 10;
  std::vector<int> data(size);
  bm_memcpy_d2s(bm_handle, data.data(), mem);
  std::cout<<"-------------------------------------"<<std::endl;
  std::cout<< data[data.size()-1] << std::endl;
  for(int i=0;i<10;i++){
    std::cout<< data[i] << std::endl;
  }
  std::cout<<"-------------------------------------"<<std::endl;
  auto ptr = data.data();
  ptr[0] = ptr[0];
}