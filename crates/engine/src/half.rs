// IEEE binary16 conversions. The storage type is `u16` because the toolchain
// has no stable half-precision float.

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits as u32) >> 10) & 0x1f;
    let mantissa = (bits as u32) & 0x3ff;
    let widened = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut exponent = -14i32;
            let mut mantissa = mantissa;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(widened)
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00 | (mantissa >> 13) as u16
        };
    }
    if exponent == 0 {
        return sign;
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let kept = (mantissa >> 13) as u16;
        let remainder = mantissa & 0x1fff;
        let mut half = sign | (((unbiased + 15) as u16) << 10) | kept;
        if remainder > 0x1000 || (remainder == 0x1000 && kept & 1 == 1) {
            half += 1;
        }
        return half;
    }
    if unbiased < -25 {
        return sign;
    }

    let shift = (-unbiased - 1) as u32;
    let significand = mantissa | 0x0080_0000;
    let kept = significand >> shift;
    let remainder = significand & ((1 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    let mut half = sign | kept as u16;
    if remainder > halfway || (remainder == halfway && kept & 1 == 1) {
        half += 1;
    }
    half
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    fn widened_reference(bits: u16) -> f64 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let mantissa = (bits & 0x3ff) as f64;
        match exponent {
            0 => sign * mantissa * 2f64.powi(-24),
            0x1f => {
                if mantissa == 0.0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
        }
    }

    #[test]
    fn known_values_round_to_nearest_even() {
        for (value, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (65519.0, 0x7bff),
            (65520.0, 0x7c00),
            (70000.0, 0x7c00),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (2f32.powi(-24), 0x0001),
            (2f32.powi(-25), 0x0000),
            (3.0 * 2f32.powi(-25), 0x0002),
            (2f32.powi(-14), 0x0400),
            (2f32.powi(-15), 0x0200),
            (6.1035156e-5, 0x0400),
            (1.0 + 2f32.powi(-11), 0x3c00),
            (1.0 + 3.0 * 2f32.powi(-11), 0x3c02),
        ] {
            assert_eq!(f32_to_f16(value), bits, "value {value:e}");
        }
        assert_eq!(f32_to_f16(f32::NAN) & 0x7c00, 0x7c00);
        assert_ne!(f32_to_f16(f32::NAN) & 0x03ff, 0);
    }

    #[test]
    fn every_half_widens_exactly_and_round_trips() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            let is_nan = (bits >> 10) & 0x1f == 0x1f && bits & 0x3ff != 0;
            if is_nan {
                assert!(value.is_nan(), "bits {bits:#06x}");
                continue;
            }
            let expected = widened_reference(bits);
            assert_eq!(
                value as f64, expected,
                "bits {bits:#06x}: expected {expected:e}, got {value:e}"
            );
            assert_eq!(f32_to_f16(value), bits, "round trip of {bits:#06x}");
        }
    }

    #[test]
    fn rounding_stays_within_half_an_ulp_and_breaks_ties_to_even() {
        let mut state = 0x1234_5678u32;
        for _ in 0..200_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = f32::from_bits(state);
            if !value.is_finite() {
                continue;
            }
            let half = f32_to_f16(value);
            let magnitude = half & 0x7fff;
            if magnitude >= 0x7bff {
                continue;
            }
            let widened = f16_to_f32(half) as f64;
            let above = f16_to_f32(magnitude + 1 | (half & 0x8000)) as f64;
            let ulp = (above - widened).abs();
            let distance = (value as f64 - widened).abs();
            assert!(
                distance <= ulp / 2.0 * (1.0 + 1e-12),
                "value {value:e} rounded to {widened:e}, {} ulps away",
                distance / ulp
            );
            if distance == ulp / 2.0 {
                assert_eq!(
                    magnitude & 1,
                    0,
                    "value {value:e} is a midpoint but {widened:e} is odd"
                );
            }
        }
    }
}
