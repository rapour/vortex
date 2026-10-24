// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A naive walk over unset bits, which [`BitBuffer`] only offers for set bits.

use vortex_buffer::BitBuffer;

/// Call `f` with the index of every unset bit, in ascending order.
pub(super) fn for_each_unset_index(bits: &BitBuffer, mut f: impl FnMut(usize)) {
    let chunks = bits.chunks();
    let mut start = 0;
    for chunk in chunks.iter() {
        for_each_set_bit(!chunk, start, &mut f);
        start += 64;
    }
    // The remainder word is zero padded, so its padding comes out set once inverted and has to be
    // masked off.
    let held = chunks.remainder_len();
    for_each_set_bit(!chunks.remainder_bits() & ((1 << held) - 1), start, &mut f);
}

/// Call `f` with `start` plus the index of every set bit in `word`.
#[inline]
fn for_each_set_bit(mut word: u64, start: usize, f: &mut impl FnMut(usize)) {
    while word != 0 {
        f(start + word.trailing_zeros() as usize);
        word &= word - 1;
    }
}
