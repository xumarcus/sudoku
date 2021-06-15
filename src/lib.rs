use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt::{self, Debug, Display};
use std::str::{self, FromStr};

use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum ParseSudokuError {
    #[error("Invalid character")]
    InvalidCharacter(u8),
    #[error("Too few characters")]
    TooFewCharacters,
    #[error("Too many characters")]
    TooManyCharacters,
}

/*
enum ConstraintType {
    Block,
    Row,
    Col,
}
*/

const N1: usize = 3;
const N2: usize = N1 * N1;
const N3: usize = N1 * N2;
const N4: usize = N2 * N2;
const S09: u128 = (1 << N2) - 1;
const S27: u128 = (1 << N3) - 1;
const ROW_INDICES: u128 = S09; // Alias
const COL_INDICES: u128 = 0x1008040201008040201;
const BLK_INDICES: u128 = 0x1c0e07;
const CONSTRAINT_INDICES: u128 = 0x100100110010011001001;

#[derive(Clone)]
pub struct Sudoku([u128; N2]);

impl Debug for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.0.iter() {
            for cell in core::ItemIterator::new(*row) {
                write!(f, "_{:09b}", cell)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Display for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.0.iter() {
            for cell in core::ItemIterator::new(*row) {
                let chr = match cell {
                    0 => '!',
                    cell if cell.next_power_of_two() == cell => {
                        ((cell.trailing_zeros() as u8) + b'1') as char
                    }
                    _ => '.',
                };
                write!(f, "{}", chr)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl FromStr for Sudoku {
    type Err = ParseSudokuError;

    #[rustfmt::skip]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let buf = s
            .bytes()
            .filter_map(|chr| match chr {
                x if x.is_ascii_whitespace() => None,
                x @ (b'1'..=b'9')  => Some(Ok(1 << (x - b'1'))),
                b'0' | b'.' | b'_' => Some(Ok(S09)),
                x                  => Some(Err(ParseSudokuError::InvalidCharacter(x))),
            })
            .collect::<Result<Vec<u128>, Self::Err>>()?;
        match buf.len().cmp(&N4) {
            Ordering::Less    => Err(ParseSudokuError::TooFewCharacters),
            Ordering::Greater => Err(ParseSudokuError::TooManyCharacters),
            Ordering::Equal   => {
                let sudoku = buf
                    .chunks_exact(N2)
                    .map(|chunk| chunk.iter().rev().fold(0u128, |acc, x| (acc << N2) + x))
                    .collect::<Vec<u128>>()
                    .try_into()
                    .expect("Should have 9x u128s");
                Ok(Sudoku(sudoku))
            }
        }
    }
}

pub struct Backtrack {
    sudoku: Sudoku,
    idx: usize,
    u9_bit_iter: core::BitIterator,
}

impl Backtrack {
    fn new(sudoku: Sudoku, idx: usize, u9_domain: u128) -> Self {
        Self {
            sudoku,
            idx,
            u9_bit_iter: core::BitIterator::new(u9_domain),
        }
    }
}

impl Iterator for Backtrack {
    type Item = Solutions;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(u9_bit) = self.u9_bit_iter.next() {
            let mut cp = self.sudoku.clone();
            let shl = N2 * (self.idx % N2);
            cp.0[self.idx / N2] &= !(S09 << shl);
            cp.0[self.idx / N2] |= u9_bit << shl;
            if let Some(_) = cp.make_consistent(1 << self.idx) {
                return Some(cp.solutions_from_consistent());
            }
        }
        None
    }
}

pub enum Solutions {
    Multiple(Box<std::iter::Flatten<Backtrack>>),
    Single(std::iter::Once<Sudoku>),
    None,
}

impl Iterator for Solutions {
    type Item = Sudoku;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Multiple(iter) => iter.next(),
            Self::Single(iter) => iter.next(),
            Self::None => None,
        }
    }
}

impl Sudoku {
    pub fn solve(self) -> Option<Self> {
        self.solutions().next()
    }

    // Returns none if no solution without guessing
    pub fn solutions(mut self) -> Solutions {
        match self.make_consistent(CONSTRAINT_INDICES) {
            Some(_) => self.solutions_from_consistent(),
            None => Solutions::None,
        }
    }

    pub fn generate() -> Self {
        unimplemented!()
    }

    fn iter(&self) -> impl Iterator<Item = u128> + '_ {
        self.0
            .iter()
            .cloned()
            .flat_map(core::NonZeroItemIterator::new)
    }

    fn solutions_from_consistent(self) -> Solutions {
        match self
            .iter()
            .map(|u9_domain| (u9_domain, u9_domain.count_ones()))
            .enumerate()
            .filter(|t| t.1 .1 > 1)
            .min_by_key(|t| t.1 .1)
        {
            Some((idx, (u9_domain, _))) => {
                Solutions::Multiple(Box::new(Backtrack::new(self, idx, u9_domain).flatten()))
            }
            None => Solutions::Single(std::iter::once(self)),
        }
    }

    fn make_consistent(&mut self, mut q: u128) -> Option<()> {
        while let Some(bit) = core::BitIterator::new(q).next() {
            let idx = bit.trailing_zeros() as usize;
            q &= !bit;
            q |= self.enforce(self.row(idx), core::row_indices(idx))?;
            q |= self.enforce(self.col(idx), core::col_indices(idx))?;
            q |= self.enforce(self.blk(idx), core::blk_indices(idx))?;
        }
        Some(())
    }

    fn row(&self, idx: usize) -> u128 {
        let rix = idx / N2;
        self.0[rix]
    }

    fn col(&self, idx: usize) -> u128 {
        let cix = idx % N2;
        let shr = N2 * cix;
        self.0
            .iter()
            .rev()
            .fold(0, |acc, x| (acc << N2) + ((x >> shr) & S09))
    }

    fn blk(&self, idx: usize) -> u128 {
        let rbx = (idx / N2) / N1;
        let cbx = (idx % N2) / N1;
        let shr = N3 * cbx;
        self.0
            .get(rbx * N1..(rbx + 1) * N1)
            .expect("Index is in range")
            .iter()
            .rev()
            .fold(0, |acc, x| (acc << N3) + ((x >> shr) & S27))
    }

    fn enforce(&mut self, buf: u128, indices: u128) -> Option<u128> {
        Some(
            core::BitIterator::new(indices)
                .zip(
                    core::NonZeroItemIterator::new(buf)
                        .zip(core::NonZeroItemIterator::new(core::prune_buf(buf)?)),
                )
                .filter(|(_, (row, row_))| row != row_)
                .fold(0, |q, (bit, (_, row_))| {
                    let idx = bit.trailing_zeros() as usize;
                    let shl = N2 * (idx % N2);
                    self.0[idx / N2] &= !(S09 << shl);
                    self.0[idx / N2] |= row_ << shl;
                    q | bit
                }),
        )
    }
}

mod core {
    use super::*;

    #[inline]
    pub fn row_indices(idx: usize) -> u128 {
        ROW_INDICES << (N2 * (idx / N2))
    }

    #[inline]
    pub fn col_indices(idx: usize) -> u128 {
        COL_INDICES << (idx % N2)
    }

    #[inline]
    pub fn blk_indices(idx: usize) -> u128 {
        let rbx = (idx / N2) / N1;
        let cbx = (idx % N2) / N1;
        BLK_INDICES << N1 * (cbx + N2 * rbx)
    }

    #[derive(Clone, Debug)]
    pub struct NonZeroItemIterator {
        st: u128,
    }

    impl NonZeroItemIterator {
        pub fn new(st: u128) -> Self {
            Self { st }
        }
    }

    impl Iterator for NonZeroItemIterator {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.st & S09;
            (item != 0).then(|| {
                self.st >>= N2;
                item
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct ItemIterator {
        st: u128,
        count: usize,
    }

    impl ItemIterator {
        pub fn new(st: u128) -> Self {
            Self { st, count: N2 }
        }
    }

    impl Iterator for ItemIterator {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            (self.count != 0).then(|| {
                let item = self.st & S09;
                self.st >>= N2;
                self.count -= 1;
                item
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct BitIterator {
        st: u128,
    }

    impl BitIterator {
        pub fn new(st: u128) -> Self {
            Self { st }
        }
    }

    impl Iterator for BitIterator {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            (self.st != 0).then(|| {
                let item = self.st & self.st.wrapping_neg();
                self.st -= item;
                item
            })
        }
    }

    fn dfs(buf: u128, vis: u128, idx: usize, rev_matching: &mut u128) -> bool {
        let u9_domain = (buf >> (N2 * idx)) & S09;
        for u9_bit in core::BitIterator::new(u9_domain) {
            let pos = u9_bit.trailing_zeros() as usize;
            let u9_rev_matched = (*rev_matching >> (N2 * pos)) & S09;
            if u9_rev_matched != 0 {
                if u9_rev_matched & vis != 0 {
                    continue;
                }
                let u9_rev_matched_pos = u9_rev_matched.trailing_zeros() as usize;
                if !dfs(buf, vis | (1 << idx), u9_rev_matched_pos, rev_matching) {
                    continue;
                }
            }
            *rev_matching &= !(S09 << (N2 * pos));
            *rev_matching |= 1 << (idx + N2 * pos);
            return true;
        }
        false
    }

    pub fn kuhn(buf: u128) -> Option<u128> {
        let mut rev_matching = 0;
        for idx in 0..N2 {
            dfs(buf, 0, idx, &mut rev_matching);
        }
        let matching = core::BitIterator::new(rev_matching)
            .map(|bit| {
                let pos = bit.trailing_zeros() as usize;
                1 << (pos / N2 + N2 * (pos % N2))
            })
            .sum::<u128>();
        (matching.count_ones() as usize == N2).then(|| matching)
    }

    pub fn prune_buf(buf: u128) -> Option<u128> {
        let mut buf_pruned = buf;
        let mut matchings = 0;
        for (idx, u9_domain) in core::NonZeroItemIterator::new(buf).enumerate() {
            let mask = S09 << (N2 * idx);
            let mut is_no_match = true;
            for u9_bit in core::BitIterator::new(u9_domain) {
                let bit = u9_bit << (N2 * idx);
                let pruned = (buf_pruned & !mask) | bit;
                if let Some(matching) = core::kuhn(pruned) {
                    matchings |= matching;
                    is_no_match = false;
                } else {
                    buf_pruned &= !bit;
                }
            }
            if is_no_match {
                return None;
            }
        }
        debug_assert_eq!(buf_pruned, matchings);
        Some(buf_pruned)
    }
}

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_81_BITS: u128 = 0x1_ffff_ffff_ffff_ffff_ffff;

    #[test]
    fn test_iterate_bits() {
        let mut it = core::BitIterator::new(0xabcdef);
        assert_eq!(it.nth(2), Some(0x04)); // 3rd
        assert_eq!(it.nth(6), Some(0x800)); // 10th
        assert_eq!(it.nth(6), Some(0x800000)); // 17th
        assert_eq!(it.next(), None);
    }

    #[quickcheck]
    fn prop_iterate_bits(st: u128) -> bool {
        let mut sum = 0;
        for bit in core::BitIterator::new(st) {
            if bit.count_ones() != 1 {
                return false;
            }
            sum += bit;
        }
        sum == st
    }

    #[test]
    fn test_iterate_items_nine_times() {
        let mut it = core::ItemIterator::new(0b0_000110110_111110011_011100010);
        assert_eq!(it.next(), Some(0b011100010));
        assert_eq!(it.next(), Some(0b111110011));
        assert_eq!(it.next(), Some(0b000110110));
        assert_eq!(it.nth(5), Some(0));
        assert_eq!(it.next(), None);
    }

    #[quickcheck]
    fn prop_iterate_items_nine_times(st: u128) -> bool {
        let st_ = st & FIRST_81_BITS;
        let mut sum = 0;
        for (i, item) in core::ItemIterator::new(st_).enumerate() {
            sum += item << (N2 * i);
        }
        sum == st_
    }

    #[test]
    fn test_iterate_nonzero_items() {
        let mut it = core::NonZeroItemIterator::new(0b0_000110110_111110011_011100010);
        assert_eq!(it.next(), Some(0b011100010));
        assert_eq!(it.next(), Some(0b111110011));
        assert_eq!(it.next(), Some(0b000110110));
        assert_eq!(it.next(), None);
    }

    #[quickcheck]
    fn prop_iterate_nonzero_items(st: u128) -> bool {
        let st_ = st & FIRST_81_BITS;
        let mut sum = 0;
        let iter = core::NonZeroItemIterator::new(st_);
        for (i, item) in iter.clone().enumerate() {
            if item == 0 {
                return false;
            }
            sum += item << (N2 * i);
        }
        sum == st_ & (FIRST_81_BITS >> (N2 * (N2 - iter.count())))
    }

    #[test]
    fn test_kuhn_one_one_matching() {
        let same_indices = 0b100000000_010000000_001000000_000100000_000010000_000001000_000000100_000000010_000000001;
        assert_eq!(core::kuhn(same_indices), Some(same_indices));

        // [3, 1, 7, 5, 8, 0, 2, 4, 6]
        let diff_indices = 0b001000000_000010000_000000100_000000001_100000000_000100000_010000000_000000010_000001000;
        assert_eq!(core::kuhn(diff_indices), Some(diff_indices));
    }

    #[test]
    fn test_kuhn_one_cell_empty() {
        let row = 0b000100000_010000000_000010000_000000100_111111111_000000001_000001000_001000000_100000000;
        let res = 0b000100000_010000000_000010000_000000100_000000010_000000001_000001000_001000000_100000000;
        assert_eq!(core::kuhn(row), Some(res));
    }

    #[test]
    fn test_kuhn_simple_sudoku() {
        let row = 0b111111111_111111111_111111111_010000000_111111111_111111111_111111111_000000010_100000000;
        //  res = 0b000000001_000000100_000001000_010000000_000010000_000100000_001000000_000000010_100000000;
        let matching = core::kuhn(row).unwrap();
        assert_eq!(row & matching, matching);
    }

    #[test]
    fn prune_buf_one_cell_empty() {
        let row = 0b000100000_010000000_000010000_000000100_111111111_000000001_000001000_001000000_100000000;
        let res = 0b000100000_010000000_000010000_000000100_000000010_000000001_000001000_001000000_100000000;
        assert_eq!(core::prune_buf(row), Some(res));
    }

    #[quickcheck]
    fn prop_row_indices_contain_self(idx: usize) -> bool {
        let idx = idx % N2;
        core::row_indices(idx) & (1 << idx) != 0
    }

    #[quickcheck]
    fn prop_col_indices_contain_self(idx: usize) -> bool {
        let idx = idx % N2;
        core::col_indices(idx) & (1 << idx) != 0
    }

    #[quickcheck]
    fn prop_blk_indices_contain_self(idx: usize) -> bool {
        let idx = idx % N2;
        core::blk_indices(idx) & (1 << idx) != 0
    }
}
