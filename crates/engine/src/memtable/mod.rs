#![allow(dead_code)]

use gbdb_common::key::SequenceNumber;

/// The result of a MemTable lookup at a snapshot.
///
/// The distinction between `Deleted` and `NotFound` is what makes the LSM read
/// path correct: a tombstone terminates the search, an absent key continues it
/// into older tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LookupResult<'a> {
    /// Key exists at this snapshot. Borrowed from the index arena.
    Found(&'a [u8]),
    /// Key was explicitly deleted (tombstone). Do not search older tables.
    Deleted,
    /// Key not present in this MemTable. Continue into older tables/SSTables.
    NotFound,
}

/// The interface all MemTable index structures must satisfy.
///
/// Keys crossing this boundary are *encoded* internal keys (see
/// `gbdb_common::key`), never structured ones — the index only ever needs to
/// compare and copy bytes, and the same encoding is what SSTables store.
///
/// Implementations must be:
/// - Ordered by `gbdb_common::key::compare`
/// - Safe for concurrent access (one writer, multiple readers is the minimum)
/// - Traversable in key order, for flush and for prefix scans
///
/// This trait is a compile-time seam for swapping the index implementation, not
/// a runtime polymorphism point. Callers should be generic over it rather than
/// holding a `dyn MemIndex`, so probes don't pay a virtual call.
pub(crate) trait MemIndex: Send + Sync {
    type Cursor<'a>: MemCursor
    where
        Self: 'a;

    /// Append a version. Never overwrites: an existing entry for the same user
    /// key at a lower sequence number stays visible to older snapshots, and
    /// deletions arrive through this same path as `RecordType::Deletion` with an
    /// empty value.
    ///
    /// Key and value are copied into the index's own storage.
    fn insert(&self, encoded_key: &[u8], value: &[u8]);

    /// Newest version of `user_key` with `sequence_number <= read_seq`.
    fn get<'a>(&'a self, user_key: &[u8], read_seq: SequenceNumber) -> LookupResult<'a>;

    /// Cursor over all entries in encoded-key order.
    fn cursor(&self) -> Self::Cursor<'_>;

    /// Approximate memory usage in bytes, used to decide when to freeze.
    fn approximate_size(&self) -> usize;

    /// Number of entries, including tombstones and shadowed versions.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Ordered traversal over a [`MemIndex`].
///
/// Positioned cursor rather than an `Iterator` of owned pairs: flush walks every
/// entry of a full MemTable, so a per-entry allocation there is a hot-loop cost.
/// `key` and `value` borrow from the index for as long as the cursor is not
/// moved.
///
/// A cursor starts invalid; call [`MemCursor::seek_to_first`] or
/// [`MemCursor::seek`] before reading.
pub(crate) trait MemCursor {
    /// Position at the smallest key. Full scans (flush) start here.
    fn seek_to_first(&mut self);

    /// Position at the first key >= `encoded_key`, or invalid if none exists.
    ///
    /// This is both the point-lookup primitive (seek to
    /// `InternalKey::seek(user_key, read_seq)`) and the prefix-scan primitive
    /// that graph adjacency walks are built on.
    fn seek(&mut self, encoded_key: &[u8]);

    /// Move to the next key in order, or become invalid at the end.
    fn advance(&mut self);

    /// Whether the cursor is positioned at an entry.
    fn is_valid(&self) -> bool;

    /// Encoded internal key at the current position.
    ///
    /// Panics if the cursor is not valid.
    fn key(&self) -> &[u8];

    /// Value at the current position; empty for tombstones.
    ///
    /// Panics if the cursor is not valid.
    fn value(&self) -> &[u8];
}
