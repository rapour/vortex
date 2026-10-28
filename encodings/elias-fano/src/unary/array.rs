// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use num_traits::AsPrimitive;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMeta;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::compress::unary_decode;

/// A [`Unary`]-encoded Vortex array.
pub type UnaryArray = Array<Unary>;

/// Metadata stored alongside the `bits` buffer.
#[derive(Clone, prost::Message)]
pub(super) struct UnaryMetadata {
    /// The number of set bits, which is also the sum of the values. With the array's length it
    /// gives the exact bit length, which the buffer's bytes alone do not.
    #[prost(uint64, tag = "1")]
    pub(super) num_ones: u64,
    /// Where the bits start in the buffer's first byte, so below 8.
    #[prost(uint32, tag = "2")]
    pub(super) offset: u32,
}

/// The parts of a [`Unary`] array: its bits.
#[derive(Clone, Debug)]
pub struct UnaryData {
    bits: BufferHandle,
    meta: BitBufferMeta,
}

impl Display for UnaryData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "bits: {}", self.meta.len())
    }
}

impl ArrayHash for UnaryData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.bits.array_hash(state, accuracy);
        self.meta.offset().hash(state);
    }
}

impl ArrayEq for UnaryData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.meta.offset() == other.meta.offset() && self.bits.array_eq(&other.bits, accuracy)
    }
}

/// The number of bits `len` values take with `num_ones` set bits, if a `usize` can count them.
pub(super) fn bit_len(len: usize, num_ones: u64) -> Option<usize> {
    u64::try_from(len)
        .ok()?
        .checked_add(num_ones)
        .and_then(|bits| usize::try_from(bits).ok())
}

/// The number of set bits in `array`, which is its bit length minus its length.
fn num_ones_of(array: ArrayView<'_, Unary>) -> VortexResult<u64> {
    Ok(u64::try_from(array.meta.len() - array.len())?)
}

impl UnaryData {
    /// Rebuild the parts from a stored buffer whose bits start `offset` bits in. Only the offset
    /// and the buffer's size are checked here. The readers rebuild the bits with
    /// [`BitBuffer::new_with_offset`], whose own check can pass a short buffer on 32-bit, so this
    /// one must be exact. [`Array::try_from_parts`] validates the rest.
    fn from_buffer(
        buffer: BufferHandle,
        num_ones: u64,
        offset: usize,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < 8,
            "UnaryArray bit offset must be below 8, got {offset}"
        );
        let bits = bit_len(len, num_ones).ok_or_else(|| {
            vortex_err!(
                "UnaryArray of {len} values with {num_ones} set bits overflows a usize bit count"
            )
        })?;
        // Saturating can only undercount what the buffer holds, so it never passes one too short.
        vortex_ensure!(
            buffer
                .len()
                .saturating_mul(8)
                .checked_sub(offset)
                .is_some_and(|room| room >= bits),
            "UnaryArray needs {bits} bits from bit {offset}, but its buffer holds {} bytes",
            buffer.len()
        );
        Ok(Self {
            bits: buffer,
            meta: BitBufferMeta::new(offset, bits),
        })
    }

    /// Check the parts against an array of `len` values of `dtype`.
    ///
    /// The dtype and the bit length are always checked. When the bits are on the host, this also
    /// checks the last bit and counts the set bits, and if the values could sum past the ptype's
    /// maximum, walks them once to check that each value fits. The readers rely on this and check
    /// no value themselves.
    fn validate(&self, dtype: &DType, len: usize) -> VortexResult<()> {
        let DType::Primitive(ptype, NonNullable) = dtype else {
            vortex_bail!("UnaryArray requires a non-nullable unsigned integer dtype, got {dtype}");
        };
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "UnaryArray requires a non-nullable unsigned integer dtype, got {dtype}"
        );
        let num_ones = self.meta.len().checked_sub(len).ok_or_else(|| {
            vortex_err!(
                "UnaryArray of {len} values has only {} bits",
                self.meta.len()
            )
        })?;
        if !self.bits.is_on_host() {
            return Ok(());
        }
        let bits = self.to_bit_buffer();
        // An unset last bit closes the last value, and leaves an empty array no bits at all.
        vortex_ensure!(
            bits.is_empty() || !bits.value(bits.len() - 1),
            "UnaryArray bits must end with an unset bit"
        );
        // The linear checks run last. `num_ones` set bits among `len + num_ones` leave exactly
        // `len` unset, one closing each value.
        vortex_ensure_eq!(bits.true_count(), num_ones);
        let max = ptype.max_value_as_u64();
        if u64::try_from(num_ones)? > max {
            let mut fits = true;
            self.for_each_value(|value| {
                fits &= u64::try_from(value).is_ok_and(|value| value <= max);
            });
            vortex_ensure!(fits, "UnaryArray holds a value that does not fit {ptype}");
        }
        Ok(())
    }

    /// The bits. Panics if the buffer is on a device.
    pub(super) fn to_bit_buffer(&self) -> BitBuffer {
        BitBuffer::new_with_offset(
            self.bits.as_host().clone(),
            self.meta.len(),
            self.meta.offset(),
        )
    }

    /// Call `f` with every value, in order, from one walk over the unset bits.
    pub(super) fn for_each_value(&self, mut f: impl FnMut(usize)) {
        let mut start = 0;
        self.to_bit_buffer().for_each_unset_index(|end| {
            f(end - start);
            start = end + 1;
        });
    }

    /// The value at `index`, found by scanning from the first bit to the unset bit before it, then
    /// on to its own closing bit.
    pub(super) fn value_at(&self, index: usize) -> VortexResult<u64> {
        let missing = || vortex_err!("UnaryArray has no unset bit for index {index}");
        let bits = self.to_bit_buffer();
        let start = match index.checked_sub(1) {
            Some(previous) => bits.select_unset(previous).ok_or_else(missing)? + 1,
            None => 0,
        };
        // The value is the run of set bits from `start` to the next unset bit, so this continues
        // from `start` rather than scanning from the first bit again.
        let value = bits.slice(start..).select_unset(0).ok_or_else(missing)?;
        Ok(u64::try_from(value)?)
    }
}

impl VTable for Unary {
    type TypedArrayData = UnaryData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.unary");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 0);
        data.validate(dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bits.clone(),
            _ => vortex_panic!("UnaryArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bits".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 1);
        let data = UnaryData::from_buffer(
            buffers[0].clone(),
            num_ones_of(array)?,
            array.meta.offset(),
            array.len(),
        )?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            ArraySlots::new(),
        ))
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            UnaryMetadata {
                num_ones: num_ones_of(array)?,
                offset: u32::try_from(array.meta.offset())?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 1);
        vortex_ensure_eq!(children.len(), 0);
        let metadata = UnaryMetadata::decode(metadata)?;
        let data = UnaryData::from_buffer(
            buffers[0].clone(),
            metadata.num_ones,
            usize::try_from(metadata.offset)?,
            len,
        )?;
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            ArraySlots::new(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("UnaryArray has no slots, got index {idx}")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(unary_decode(&array).into_array()))
    }
}

impl OperationsVTable<Unary> for Unary {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Unary>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = array.value_at(index)?;
        // `validate` keeps every value within the ptype.
        Ok(match_each_unsigned_integer_ptype!(
            array.dtype().as_ptype(),
            |T| { Scalar::primitive::<T>(value.as_(), NonNullable) }
        ))
    }
}

impl ValidityVTable<Unary> for Unary {
    fn validity(_array: ArrayView<'_, Unary>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

/// Unary encoding for non-nullable unsigned integers, in any order.
///
/// The values are stored as one bit array. Each value `v` is written in unary, as `v` set bits
/// followed by an unset bit that closes it, so the array takes `len + num_ones` bits, where
/// `num_ones`, the number of set bits, is the sum of the values. The value at index `k` is the
/// number of set bits between the `k`-th unset bit and the one before it.
///
/// For `[2, 0, 3]` the bits are `110 0 1110`.
///
/// Reads are deliberately simple for now: `scalar_at` scans from the first bit, and a slice
/// decodes the whole array.
#[derive(Clone, Debug)]
pub struct Unary;

impl Unary {
    /// Assemble an array of `len` values from its bits.
    ///
    /// Prefer [`unary_encode`](crate::unary_encode) unless you already hold the bits.
    pub fn try_new(bits: BitBuffer, ptype: PType, len: usize) -> VortexResult<UnaryArray> {
        let dtype = DType::Primitive(ptype, NonNullable);
        // `shrink_offset` drops the whole bytes before and after the bits, so the buffer holds only
        // the bytes they use and the offset is below 8, as `BitBufferMeta` requires.
        let (offset, bits_len, buffer) = bits.shrink_offset().into_inner();
        let data = UnaryData {
            bits: BufferHandle::new_host(buffer),
            meta: BitBufferMeta::new(offset, bits_len),
        };
        Array::try_from_parts(ArrayParts::new(Unary, dtype, len, data, ArraySlots::new()))
    }
}
