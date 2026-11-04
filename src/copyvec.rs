//! `CopyVec`: a fixed-capacity list stored inline, `Copy` when its items are.
//! Battle state uses it instead of `Vec` / `SmallVec` so a state copy is a
//! plain `memcpy` (no heap, no drop glue).

use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy)]
pub struct CopyVec<T: Copy, const N: usize> {
    len: u8,
    /// `items[..len]` are initialized.
    items: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> CopyVec<T, N> {
    pub fn new() -> Self {
        const { assert!(N <= u8::MAX as usize) };
        CopyVec { len: 0, items: [MaybeUninit::uninit(); N] }
    }

    /// Append an item. Panics when full: every list has a bound from the
    /// battle rules (slots, team size), and a panic shows a wrong bound.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, x: T) {
        assert!((self.len as usize) < N, "CopyVec full (capacity {})", N);
        self.items[self.len as usize] = MaybeUninit::new(x);
        self.len += 1;
    }

    /// Append an item; when full, drop the oldest one first.
    #[inline]
    pub fn push_evict(&mut self, x: T) {
        if self.len as usize == N {
            self.remove(0);
        }
        self.push(x);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: the item was below `len`.
        Some(unsafe { self.items[self.len as usize].assume_init() })
    }

    pub fn remove(&mut self, i: usize) -> T {
        let n = self.len as usize;
        assert!(i < n, "CopyVec::remove index {} out of range (len {})", i, n);
        let x = self[i];
        self.items.copy_within(i + 1..n, i);
        self.len -= 1;
        x
    }

    pub fn insert(&mut self, i: usize, x: T) {
        let n = self.len as usize;
        assert!(i <= n && n < N, "CopyVec::insert at {} (len {}, capacity {})", i, n, N);
        self.items.copy_within(i..n, i + 1);
        self.items[i] = MaybeUninit::new(x);
        self.len += 1;
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut w = 0;
        for r in 0..self.len as usize {
            if keep(&self[r]) {
                self.items[w] = self.items[r];
                w += 1;
            }
        }
        self.len = w as u8;
    }

    pub fn truncate(&mut self, n: usize) {
        if n < self.len as usize {
            self.len = n as u8;
        }
    }
}

impl<T: Copy, const N: usize> Default for CopyVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> Deref for CopyVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: `items[..len]` are initialized; `MaybeUninit<T>` has `T`'s layout.
        unsafe { std::slice::from_raw_parts(self.items.as_ptr() as *const T, self.len as usize) }
    }
}

impl<T: Copy, const N: usize> DerefMut for CopyVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`.
        unsafe { std::slice::from_raw_parts_mut(self.items.as_mut_ptr() as *mut T, self.len as usize) }
    }
}

impl<T: Copy + std::fmt::Debug, const N: usize> std::fmt::Debug for CopyVec<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for CopyVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Copy, const N: usize> From<&[T]> for CopyVec<T, N> {
    fn from(s: &[T]) -> Self {
        let mut v = Self::new();
        for &x in s {
            v.push(x);
        }
        v
    }
}

impl<T: Copy, const N: usize> From<Vec<T>> for CopyVec<T, N> {
    fn from(v: Vec<T>) -> Self {
        Self::from(&v[..])
    }
}

impl<T: Copy, const N: usize> Extend<T> for CopyVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        for x in it {
            self.push(x);
        }
    }
}

impl<T: Copy, const N: usize> std::iter::FromIterator<T> for CopyVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        let mut v = Self::new();
        v.extend(it);
        v
    }
}

impl<T: Copy, const N: usize> IntoIterator for CopyVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { v: self, i: 0 }
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a CopyVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// By-value iterator over a `CopyVec`.
pub struct IntoIter<T: Copy, const N: usize> {
    v: CopyVec<T, N>,
    i: usize,
}

impl<T: Copy, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let x = self.v.get(self.i).copied();
        self.i += 1;
        x
    }
}
