//! Internal key encoding.
//!
//! An internal key is a user key plus the version metadata that makes MVCC work:
//! a sequence number and a record type. Keys are stored in their *encoded* form —
//! a single byte slice — so that the MemTable arena, SSTable blocks, and the WAL
//! all speak the same representation and share one comparator.
//!
//! Layout:
//!
//! ```text
//! [ user_key (variable) ][ !sequence_number (8B, big-endian) ][ record_type (1B) ]
//! ```
//!
//! The sequence number is stored bitwise-inverted so that ordering the fixed-size
//! trailer ascending yields sequence numbers descending — newest version first.
//!
//! Note that ordering is **not** a plain `memcmp` over the whole encoded key: with
//! variable-length user keys, the trailer of a short key would be compared against
//! the content of a longer one (`"a"` vs `"ab"`). [`compare`] splits the fixed-size
//! trailer off first, which is what makes prefix relationships sort correctly.

use std::cmp::Ordering;
use std::fmt;

/// Monotonically increasing version stamp, assigned per mutation.
pub type SequenceNumber = u64;

/// Bytes appended after the user key: inverted sequence number plus record type.
pub const TRAILER_LEN: usize = 9;

/// What a record asserts about its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RecordType {
    Value = 0,
    Deletion = 1,
}

impl RecordType {
    fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(RecordType::Value),
            1 => Some(RecordType::Deletion),
            _ => None,
        }
    }
}

/// The structured view of an internal key.
///
/// This exists to build and inspect encoded keys; it is not what the index stores.
/// Skiplist nodes, SSTable blocks, and WAL payloads all hold the encoded bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InternalKey<'a> {
    pub user_key: &'a [u8],
    pub sequence_number: SequenceNumber,
    pub record_type: RecordType,
}

impl<'a> InternalKey<'a> {
    pub fn new(
        user_key: &'a [u8],
        sequence_number: SequenceNumber,
        record_type: RecordType,
    ) -> Self {
        Self {
            user_key,
            sequence_number,
            record_type,
        }
    }

    /// The key to pass to a lower-bound seek when looking up `user_key` as of
    /// `read_seq`.
    ///
    /// Because versions sort newest-first, the first entry at or after this key is
    /// the newest version with `sequence_number <= read_seq`. `Value` is used as
    /// the record type because it produces the smallest trailer for a given
    /// sequence number, so a tombstone at exactly `read_seq` still sorts after it
    /// and is therefore found.
    pub fn seek(user_key: &'a [u8], read_seq: SequenceNumber) -> Self {
        Self::new(user_key, read_seq, RecordType::Value)
    }

    pub fn encoded_len(&self) -> usize {
        self.user_key.len() + TRAILER_LEN
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        out.reserve(self.encoded_len());
        out.extend_from_slice(self.user_key);
        out.extend_from_slice(&(!self.sequence_number).to_be_bytes());
        out.push(self.record_type as u8);
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        self.encode_into(&mut out);
        out
    }

    /// Parse an encoded key. Use this at trust boundaries — recovering from the
    /// WAL, reading an SSTable block. Within the engine, prefer [`user_key`] and
    /// [`compare`], which assume well-formed input.
    pub fn decode(encoded: &'a [u8]) -> Result<Self, KeyError> {
        if encoded.len() < TRAILER_LEN {
            return Err(KeyError::TooShort(encoded.len()));
        }
        let (user_key, trailer) = encoded.split_at(encoded.len() - TRAILER_LEN);
        let inverted = u64::from_be_bytes(trailer[..8].try_into().expect("8 bytes"));
        let record_type =
            RecordType::from_byte(trailer[8]).ok_or(KeyError::BadRecordType(trailer[8]))?;
        Ok(Self {
            user_key,
            sequence_number: !inverted,
            record_type,
        })
    }
}

impl Ord for InternalKey<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.user_key
            .cmp(other.user_key)
            // Higher sequence number = more recent = sorts first.
            .then_with(|| other.sequence_number.cmp(&self.sequence_number))
            .then_with(|| (self.record_type as u8).cmp(&(other.record_type as u8)))
    }
}

impl PartialOrd for InternalKey<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Split an encoded key into its user key and trailer.
///
/// Malformed input is clamped rather than panicking in release builds; encoded
/// keys are produced internally and corruption is caught upstream by WAL record
/// CRCs and SSTable block checksums.
#[inline]
fn split_trailer(encoded: &[u8]) -> (&[u8], &[u8]) {
    debug_assert!(
        encoded.len() >= TRAILER_LEN,
        "malformed internal key: {} bytes, need at least {TRAILER_LEN}",
        encoded.len()
    );
    encoded.split_at(encoded.len().saturating_sub(TRAILER_LEN))
}

/// The user key portion of an encoded internal key.
#[inline]
pub fn user_key(encoded: &[u8]) -> &[u8] {
    split_trailer(encoded).0
}

/// Total order over encoded internal keys: user key ascending, then sequence
/// number descending.
///
/// This is *the* comparator — the MemTable index, SSTable block layout, and any
/// merge iterator must all use it, or the on-disk order will not match the
/// in-memory order.
#[inline]
pub fn compare(a: &[u8], b: &[u8]) -> Ordering {
    let (a_user, a_trailer) = split_trailer(a);
    let (b_user, b_trailer) = split_trailer(b);
    match a_user.cmp(b_user) {
        // The trailer holds an inverted sequence number, so ascending byte order
        // over the trailer is descending sequence number.
        Ordering::Equal => a_trailer.cmp(b_trailer),
        unequal => unequal,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyError {
    TooShort(usize),
    BadRecordType(u8),
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyError::TooShort(len) => {
                write!(f, "internal key too short: {len} bytes, need {TRAILER_LEN}")
            }
            KeyError::BadRecordType(byte) => write!(f, "unknown record type: {byte}"),
        }
    }
}

impl std::error::Error for KeyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(user: &[u8], seq: SequenceNumber, ty: RecordType) -> Vec<u8> {
        InternalKey::new(user, seq, ty).encode()
    }

    #[test]
    fn round_trips() {
        let cases = [
            (&b""[..], 0, RecordType::Value),
            (&b"a"[..], 1, RecordType::Deletion),
            (&b"vertex:42:edge:out"[..], u64::MAX, RecordType::Value),
            (&[0xff; 64][..], 1 << 40, RecordType::Deletion),
        ];
        for (user, seq, ty) in cases {
            let bytes = encoded(user, seq, ty);
            assert_eq!(bytes.len(), user.len() + TRAILER_LEN);
            let decoded = InternalKey::decode(&bytes).expect("decodes");
            assert_eq!(decoded, InternalKey::new(user, seq, ty));
            assert_eq!(user_key(&bytes), user);
        }
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(
            InternalKey::decode(&[0; TRAILER_LEN - 1]),
            Err(KeyError::TooShort(TRAILER_LEN - 1))
        );
        let mut bytes = encoded(b"k", 1, RecordType::Value);
        *bytes.last_mut().unwrap() = 9;
        assert_eq!(InternalKey::decode(&bytes), Err(KeyError::BadRecordType(9)));
    }

    #[test]
    fn newer_versions_sort_first() {
        let old = encoded(b"k", 1, RecordType::Value);
        let new = encoded(b"k", 2, RecordType::Value);
        assert_eq!(compare(&new, &old), Ordering::Less);
    }

    #[test]
    fn user_keys_sort_ascending() {
        // A trailing user-key byte must beat the trailer of a shorter key, which
        // is exactly the case a naive memcmp over the whole encoding gets wrong.
        let short = encoded(b"a", 1, RecordType::Value);
        let long = encoded(b"ab", u64::MAX, RecordType::Value);
        assert_eq!(compare(&short, &long), Ordering::Less);

        let empty = encoded(b"", 0, RecordType::Value);
        assert_eq!(compare(&empty, &short), Ordering::Less);
    }

    #[test]
    fn encoded_order_matches_structured_order() {
        let users: [&[u8]; 6] = [b"", b"a", b"ab", b"b", b"\x00", b"\xff"];
        let seqs = [0u64, 1, 255, 256, u64::MAX];
        let types = [RecordType::Value, RecordType::Deletion];

        let mut keys: Vec<InternalKey<'_>> = Vec::new();
        for user in users {
            for seq in seqs {
                for ty in types {
                    keys.push(InternalKey::new(user, seq, ty));
                }
            }
        }

        for a in &keys {
            for b in &keys {
                assert_eq!(
                    compare(&a.encode(), &b.encode()),
                    a.cmp(b),
                    "encoded order disagrees with structured order for {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn seek_key_finds_newest_visible_version() {
        let mut keys: Vec<Vec<u8>> = vec![
            encoded(b"k", 10, RecordType::Value),
            encoded(b"k", 7, RecordType::Deletion),
            encoded(b"k", 3, RecordType::Value),
            encoded(b"z", 1, RecordType::Value),
        ];
        keys.sort_by(|a, b| compare(a, b));

        // Lower-bound seek at read_seq = 8 must land on the seq=7 tombstone.
        let target = InternalKey::seek(b"k", 8).encode();
        let found = keys
            .iter()
            .find(|k| compare(k, &target) != Ordering::Less)
            .expect("a version at or after the seek key");
        let decoded = InternalKey::decode(found).unwrap();
        assert_eq!(decoded.sequence_number, 7);
        assert_eq!(decoded.record_type, RecordType::Deletion);

        // A seek at exactly a version's sequence number must find that version,
        // tombstone or not.
        let target = InternalKey::seek(b"k", 7).encode();
        let found = keys
            .iter()
            .find(|k| compare(k, &target) != Ordering::Less)
            .expect("a version at or after the seek key");
        assert_eq!(InternalKey::decode(found).unwrap().sequence_number, 7);
    }
}
