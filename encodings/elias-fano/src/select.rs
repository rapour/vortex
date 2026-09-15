// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Windowed rank/select over the upper bit array.
//!
//! Decoding needs the `nth` set bit and the `nth` unset bit within a half-open bit range, so a
//! sample can give a lower bound and the scan can start from it.
//!
//! Both read the backing bytes directly, so a lookup allocates nothing. [`params::LOG_SAMPLING0`]
//! and [`params::LOG_SAMPLING1`] cap a window at 512 unset or 256 set bits — eight to sixteen
//! words — which a scalar walk covers without vectorising.
//!
//! [`params::LOG_SAMPLING0`]: crate::params::LOG_SAMPLING0
//! [`params::LOG_SAMPLING1`]: crate::params::LOG_SAMPLING1

use vortex_buffer::BitBuffer;

/// Returns the position of the `nth` set bit within the bit range `[start, end)` of `buf`, relative
/// to `start`, or `None` if the range holds `nth` set bits or fewer.
///
/// # Panics
///
/// Panics if `start > end` or `end > buf.len()`.
#[inline]
pub fn select_range(buf: &BitBuffer, start: usize, end: usize, nth: usize) -> Option<usize> {
    select_in_range(buf, start, end, nth, false)
}

/// Returns the position of the `nth` unset bit within the bit range `[start, end)` of `buf`,
/// relative to `start`, or `None` if the range holds `nth` unset bits or fewer.
///
/// The complement of [`select_range`]: a bucket boundary in the upper array is a zero, so this is
/// what turns a high part into a rank.
///
/// # Panics
///
/// Panics if `start > end` or `end > buf.len()`.
#[inline]
pub fn select_zero_range(buf: &BitBuffer, start: usize, end: usize, nth: usize) -> Option<usize> {
    select_in_range(buf, start, end, nth, true)
}

/// Shared walk behind [`select_range`] (`zeros == false`) and [`select_zero_range`]
/// (`zeros == true`).
fn select_in_range(
    buf: &BitBuffer,
    start: usize,
    end: usize,
    nth: usize,
    zeros: bool,
) -> Option<usize> {
    assert!(start <= end, "start {start} exceeds end {end}");
    assert!(end <= buf.len(), "end {end} exceeds len {}", buf.len());

    // The window begins `offset + start` bits into the backing bytes.
    let bytes = buf.inner().as_slice();
    let base = buf.offset() + start;
    let total = end - start;

    let mut remaining = nth;
    let mut pos = 0usize;
    while pos < total {
        let width = (total - pos).min(64);
        let mut word = load_bits(bytes, base + pos, width);
        if zeros {
            // Complementing turns the padding above `width` into ones, so mask it off again.
            word = mask_to(!word, width);
        }
        let count = word.count_ones() as usize;
        if remaining < count {
            return Some(pos + select_in_word(word, remaining));
        }
        remaining -= count;
        pos += width;
    }
    None
}

/// Reads `width` bits starting at absolute bit `at`, returned in the low bits with the rest zero.
///
/// An unaligned 64 bits straddles nine bytes: the common path takes the first eight as one
/// little-endian word and folds the ninth in across the shift. Within nine bytes of the end the
/// slower path assembles whatever is there, since the bytes it cannot read land above `width`.
#[inline]
fn load_bits(bytes: &[u8], at: usize, width: usize) -> u64 {
    debug_assert!(width > 0 && width <= 64, "width {width} out of range");

    let first = at / 8;
    let shift = at % 8;

    let head = bytes.get(first..).and_then(<[u8]>::first_chunk::<8>);
    let word = match (head, bytes.get(first + 8)) {
        (Some(chunk), Some(&straddle)) => {
            let lo = u64::from_le_bytes(*chunk);
            // Shifting a `u64` by 64 is undefined, so the aligned case needs its own arm.
            if shift == 0 {
                lo
            } else {
                (lo >> shift) | (u64::from(straddle) << (64 - shift))
            }
        }
        _ => {
            let mut raw = 0u128;
            for (i, &byte) in bytes.iter().skip(first).take(9).enumerate() {
                raw |= u128::from(byte) << (8 * i);
            }
            (raw >> shift) as u64
        }
    };
    mask_to(word, width)
}

/// Clears every bit at or above `width`.
#[inline]
fn mask_to(word: u64, width: usize) -> u64 {
    if width == 64 {
        word
    } else {
        word & ((1u64 << width) - 1)
    }
}

/// Returns the index of the `nth` set bit of `word`, which must hold more than `nth` set bits.
///
/// Narrows a byte at a time first, so [`select_in_byte`] loops at most seven times.
#[inline]
fn select_in_word(word: u64, nth: usize) -> usize {
    debug_assert!(
        nth < word.count_ones() as usize,
        "rank {nth} is not present in the word"
    );

    let mut remaining = nth;
    let mut rest = word;
    let mut shift = 0usize;
    loop {
        let byte = (rest & 0xFF) as u8;
        let count = byte.count_ones() as usize;
        if remaining < count {
            return shift + select_in_byte(byte, remaining);
        }
        remaining -= count;
        rest >>= 8;
        shift += 8;
    }
}

/// Returns the index of the `nth` set bit of `byte`.
///
/// `byte & (byte - 1)` clears the lowest set bit, so `nth` of those leave the target lowest.
#[inline]
fn select_in_byte(byte: u8, nth: usize) -> usize {
    let mut rest = byte;
    for _ in 0..nth {
        rest &= rest - 1;
    }
    rest.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;

    use super::*;

    /// A pattern near 50% density, deliberately not byte-periodic.
    fn mixed_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(37) ^ 0x5A)
            .collect()
    }

    fn bit_buffer(bytes: Vec<u8>, offset: usize, len: usize) -> BitBuffer {
        BitBuffer::new_with_offset(ByteBuffer::copy_from(bytes), len, offset)
    }

    /// Positions of the bits equal to `want`, one at a time, as the reference answer.
    fn naive_positions(buf: &BitBuffer, start: usize, end: usize, want: bool) -> Vec<usize> {
        (start..end)
            .filter(|&i| buf.value(i) == want)
            .map(|i| i - start)
            .collect()
    }

    /// Both variants must agree with the bit-at-a-time reference across the whole rank range, and
    /// both must report `None` one past the last rank.
    fn check_against_naive(buf: &BitBuffer, start: usize, end: usize) {
        for (want, select) in [
            (
                true,
                select_range as fn(&BitBuffer, usize, usize, usize) -> Option<usize>,
            ),
            (
                false,
                select_zero_range as fn(&BitBuffer, usize, usize, usize) -> Option<usize>,
            ),
        ] {
            let expected = naive_positions(buf, start, end, want);
            for (nth, &expected_pos) in expected.iter().enumerate() {
                assert_eq!(
                    select(buf, start, end, nth),
                    Some(expected_pos),
                    "want={want} start={start} end={end} nth={nth}"
                );
            }
            assert_eq!(
                select(buf, start, end, expected.len()),
                None,
                "want={want} start={start} end={end} past-the-end rank"
            );
        }
    }

    #[rstest]
    #[case(0, 0, 128)]
    #[case(3, 0, 100)]
    #[case(7, 0, 50)]
    #[case(0, 0, 1)]
    #[case(0, 0, 64)]
    #[case(1, 0, 64)]
    #[case(0, 0, 65)]
    #[case(3, 0, 256)]
    #[case(0, 0, 512)]
    #[case(0, 0, 513)]
    #[case(5, 0, 1024)]
    // Windows that start partway in, which is the shape a sample lower bound produces.
    #[case(0, 1, 128)]
    #[case(0, 63, 128)]
    #[case(0, 64, 200)]
    #[case(0, 65, 200)]
    #[case(3, 70, 300)]
    #[case(5, 511, 1024)]
    // Sub-word windows, where the only word's valid width is what the zero path must respect.
    #[case(1, 0, 1)]
    #[case(1, 2, 8)]
    #[case(4, 3, 6)]
    #[case(7, 0, 1)]
    #[case(0, 9, 17)]
    #[case(2, 71, 71)]
    fn select_agrees_with_naive(#[case] offset: usize, #[case] start: usize, #[case] end: usize) {
        let buf = bit_buffer(mixed_bytes((offset + end).div_ceil(8) + 1), offset, end);
        check_against_naive(&buf, start, end);
    }

    /// 50% density is exactly where confusing ones with zeros is least visible; these make it
    /// obvious.
    #[rstest]
    #[case::all_zero(0x00)]
    #[case::all_one(0xFF)]
    #[case::sparse(0x01)]
    #[case::dense(0xFE)]
    fn select_uniform_density(#[case] fill: u8) {
        for (offset, start, end) in [
            (0usize, 0usize, 8usize),
            (0, 0, 128),
            (3, 2, 5),
            (5, 7, 130),
            (1, 200, 517),
        ] {
            let buf = bit_buffer(vec![fill; (offset + end).div_ceil(8) + 1], offset, end);
            check_against_naive(&buf, start, end);
        }
    }

    #[test]
    fn select_degenerate_buffers() {
        // All ones: no zero to find, at any rank.
        let ones = bit_buffer(vec![0xFF; 17], 0, 128);
        assert_eq!(select_zero_range(&ones, 0, 128, 0), None);
        assert_eq!(select_range(&ones, 0, 128, 127), Some(127));

        // All zeros: the nth zero is at position n, and no set bit exists.
        let zeros = bit_buffer(vec![0x00; 17], 0, 128);
        for nth in 0..128 {
            assert_eq!(
                select_zero_range(&zeros, 0, 128, nth),
                Some(nth),
                "nth={nth}"
            );
        }
        assert_eq!(select_zero_range(&zeros, 0, 128, 128), None);
        assert_eq!(select_range(&zeros, 0, 128, 0), None);
    }

    /// An empty window holds nothing, whatever it is asked for.
    #[test]
    fn select_empty_window() {
        let buf = bit_buffer(mixed_bytes(16), 0, 128);
        assert_eq!(select_range(&buf, 64, 64, 0), None);
        assert_eq!(select_zero_range(&buf, 64, 64, 0), None);
    }

    /// The counts a window reports must match `count_range`, and the first absent rank is the one
    /// at that count.
    #[test]
    fn select_count_agrees_with_count_range() {
        let buf = bit_buffer(mixed_bytes(300), 3, 2000);
        for (start, end) in [(0usize, 2000usize), (5, 1999), (7, 8), (64, 583)] {
            let ones = buf.count_range(start, end);
            let zeros = (end - start) - ones;
            if let Some(last) = ones.checked_sub(1) {
                assert!(select_range(&buf, start, end, last).is_some());
            }
            assert_eq!(select_range(&buf, start, end, ones), None);
            if let Some(last) = zeros.checked_sub(1) {
                assert!(select_zero_range(&buf, start, end, last).is_some());
            }
            assert_eq!(select_zero_range(&buf, start, end, zeros), None);
        }
    }
}
