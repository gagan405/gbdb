#![allow(dead_code)]
use bytes::Bytes;

/// Sequence number type. Monotonically increasing, assigned per write.
pub type SequenceNumber = u64;

/// The type of a record stored in the MemTable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordType {
    Value,
    Deletion,
}

/// A key as stored internally — user key + sequence number + record type.
/// Ordering: user_key ASC, sequence_number DESC (newest first for same key).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalKey {
    pub user_key: Bytes,
    pub sequence_number: SequenceNumber,
    pub record_type: RecordType,
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.user_key
            .cmp(&other.user_key)
            // Reverse sequence number order: higher seq = more recent = comes first
            .then(other.sequence_number.cmp(&self.sequence_number))
    }
}

/// The result of a MemTable lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupResult {
    /// Key exists and has a value.
    Found(Bytes),
    /// Key was explicitly deleted (tombstone). Do not search further.
    Deleted,
    /// Key not present in this MemTable. Continue search in older tables/SSTables.
    NotFound,
}

/// The interface all MemTable index structures must satisfy.
///
/// Implementations must be:
/// - Ordered by `InternalKey` (user_key ASC, seq DESC)
/// - Safe for concurrent access (one writer, multiple readers is the minimum)
/// - Iterable in key order (required for SSTable flush)
pub(crate) trait MemIndex: Send + Sync {
    /// Insert or overwrite a key-value pair.
    fn insert(&self, key: InternalKey, value: Bytes);

    /// Insert a tombstone (deletion marker) for a key.
    fn delete(&self, key: InternalKey);

    /// Look up a user key as of a given sequence number.
    /// Returns the newest version of the key with sequence_number <= read_seq.
    fn get(&self, user_key: &Bytes, read_seq: SequenceNumber) -> LookupResult;

    /// Iterate all entries in InternalKey order.
    /// Used during MemTable flush to write a sorted SSTable.
    fn iter(&self) -> Box<dyn Iterator<Item = (InternalKey, Bytes)> + '_>;

    /// Approximate memory usage in bytes. Used to decide when to freeze.
    fn approximate_size(&self) -> usize;

    /// Number of entries (including tombstones).
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
