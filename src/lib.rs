use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt::{self, Display};
use std::str::{self, FromStr};

use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum ParseSudokuError {
    #[error("Invalid character")]
    InvalidCharacter,
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
const S09: u128 = 1 << N2 - 1;
const S27: u128 = 1 << N3 - 1;
const ROW_INDICES: u128 = S09; // Alias
const COL_INDICES: u128 = 0x1008040201008040201;
const BLK_INDICES: u128 = 0x1c0e07;
const CONSTRAINT_INDICES: u128 = 0x100100110010011001001;

#[derive(Clone, Debug)]
pub struct Sudoku([u128; N2]);

impl FromStr for Sudoku {
    type Err = ParseSudokuError;

    #[rustfmt::skip]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let buf = s
            .bytes()
            .filter_map(|chr| match chr {
                b' ' | b'\n'       => None,
                x @ (b'1'..=b'9')  => Some(Ok(1 << (x - b'1'))),
                b'0' | b'.' | b'_' => Some(Ok(0)),
                _                  => Some(Err(ParseSudokuError::InvalidCharacter)),
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

impl Display for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.0.iter() {
            for cell in core::ItemIterator::new(*row) {
                let byte = match cell {
                    0 => b'!',
                    cell if cell.next_power_of_two() == cell => {
                        (cell.trailing_zeros() as u8) + b'1'
                    }
                    _ => b'.',
                };
                write!(f, "{}", byte)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

pub struct Backtrack {
    sudoku: Sudoku,
    idx: usize,
    iter: core::BitIterator,
}

impl Backtrack {
    fn new(sudoku: Sudoku, idx: usize, bits: u128) -> Self {
        Self {
            sudoku,
            idx,
            iter: core::BitIterator::new(bits),
        }
    }
}

impl Iterator for Backtrack {
    type Item = Solutions;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(bit) = self.iter.next() {
            let mut cp = self.sudoku.clone();
            let shl = N2 * (self.idx % N2);
            cp.0[self.idx / N2] &= !(S09 << shl);
            cp.0[self.idx / N2] |= bit << shl;
            if let Some(_) = cp.make_consistent(1u128 << self.idx) {
                return Some(cp.solutions_from_consistent());
            }
        }
        None
    }
}

pub enum Solutions {
    Multiple(Box<std::iter::Flatten<Backtrack>>),
    Single(std::iter::Once<Sudoku>),
}

impl Iterator for Solutions {
    type Item = Sudoku;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Multiple(iter) => iter.next(),
            Self::Single(iter) => iter.next(),
        }
    }
}

impl Sudoku {
    pub fn solve(self) -> Option<Self> {
        self.solutions().as_mut().and_then(Solutions::next)
    }

    // Returns none if no solution without guessing
    pub fn solutions(mut self) -> Option<Solutions> {
        self.make_consistent(CONSTRAINT_INDICES)
            .map(|_| self.solutions_from_consistent())
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
            .map(|bits| (bits, bits.count_ones()))
            .enumerate()
            .filter(|t| t.1 .1 > 1)
            .min_by_key(|t| t.1 .1)
        {
            Some((idx, (bits, _))) => {
                Solutions::Multiple(Box::new(Backtrack::new(self, idx, bits).flatten()))
            }
            None => Solutions::Single(std::iter::once(self)),
        }
    }

    fn make_consistent(&mut self, q: u128) -> Option<()> {
        if q != 0 {
            let mut q_ = 0;
            for bit in core::BitIterator::new(q) {
                let idx = bit.trailing_zeros() as usize;
                q_ |= self.enforce_row(idx)?;
                q_ |= self.enforce_col(idx)?;
                q_ |= self.enforce_blk(idx)?;
            }
            self.make_consistent(q_)?;
        }
        Some(())
    }

    fn enforce_row(&mut self, idx: usize) -> Option<u128> {
        let rix = idx / N2;
        let row = self.0[rix];
        self.enforce(row, ROW_INDICES << (N2 * rix))
    }

    fn enforce_col(&mut self, idx: usize) -> Option<u128> {
        let cix = idx % N2;
        let shr = N2 * cix;
        let col = self
            .0
            .iter()
            .rev()
            .fold(0, |acc, x| (acc << N2) + (x >> shr) & S09);
        self.enforce(col, COL_INDICES << cix)
    }

    fn enforce_blk(&mut self, idx: usize) -> Option<u128> {
        let rbx = idx / N2 / N1;
        let cbx = idx % N2 / N1;
        let shr = N3 * cbx;
        let blk = self
            .0
            .get(rbx * N1..(rbx + 1) * N1)
            .expect("Index is in range")
            .iter()
            .rev()
            .fold(0, |acc, x| (acc << N3) + (x >> shr) & S27);
        self.enforce(blk, BLK_INDICES << N2 * (cbx + N1 * rbx))
    }

    fn enforce(&mut self, buf: u128, indices: u128) -> Option<u128> {
        // Update buf online
        let mut buf_ = buf;
        let mut matchings = 0;
        for idx in 0..N2 {
            let mask = S09 << (N2 * idx);
            let selection = buf_ & mask;
            let is_empty = core::BitIterator::new(selection).all(|bit| {
                if matchings & bit == 0 {
                    let pruned = (buf_ & !mask) | bit;
                    if let Some(matching) = core::kuhn(pruned) {
                        matchings |= matching;
                    } else {
                        buf_ &= !bit;
                        return true;
                    }
                }
                false
            });
            if is_empty {
                return None;
            }
        }
        debug_assert_eq!(buf, matchings);
        Some(
            core::BitIterator::new(indices)
                .zip(core::NonZeroItemIterator::new(buf).zip(core::NonZeroItemIterator::new(buf_)))
                .filter_map(|(bit, (row, row_))| (row != row_).then(|| bit))
                .sum::<u128>(),
        )
    }
}

mod core {
    use super::*;

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
            (self.st != 0).then(|| {
                let item = self.st & S09;
                self.st >>= N2;
                item
            })
        }
    }

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
        for bit in core::BitIterator::new(u9_domain) {
            let pos = bit.trailing_zeros() as usize;
            let u9_rm = (*rev_matching >> (N2 * pos)) & S09;
            if u9_rm == 0 || (u9_rm & vis == 0 && dfs(buf, vis | (1 << idx), pos, rev_matching)) {
                *rev_matching &= !(S09 << (N2 * pos));
                *rev_matching |= bit << (N2 * pos);
                return true;
            }
        }
        false
    }

    pub fn kuhn(buf: u128) -> Option<u128> {
        let mut rev_matching = 0;
        for idx in 0..N2 {
            dfs(buf, 0, idx, &mut rev_matching);
        }
        let mut matching = 0;
        for (idx, bit) in core::ItemIterator::new(rev_matching).enumerate() {
            if bit != 0 {
                let pos = bit.trailing_zeros() as usize;
                matching |= 1 << (idx + N2 * pos);
            } else {
                return None;
            }
        }
        Some(matching)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_sudoku() {
        let sudoku: Sudoku = "\
92...8...
4.3.6289.
8.1..372.
..4...1..
.8..2..79
2.....53.
3.21.9.8.
.6...4913
197.3...4
"
        .parse()
        .unwrap();
        assert_eq!(
            sudoku.solve().unwrap().to_string(),
            "\
926718345
473562891
851493726
734985162
685321479
219647538
342159687
568274913
197836254
"
        );
    }

    #[test]
    fn inferrable_sudoku() {
        let sudoku: Sudoku = "\
.9....83.
..36...4.
..8......
2...9....
...438...
9.5.6....
...7..9..
.....4..6
17......5
"
        .parse()
        .unwrap();
        assert_eq!(
            sudoku.solve().unwrap().to_string(),
            "\
697245831
513689247
428371659
234597168
761438592
985162374
842756913
359814726
176923485
"
        );
    }

    #[test]
    fn minimal_sudoku() {
        let sudoku: Sudoku = "\
...7.....
1........
...43.2..
........6
...5.9...
......418
....81...
..2....5.
.4....3..
"
        .parse()
        .unwrap();
        assert_eq!(
            sudoku.solve().unwrap().to_string(),
            "\
264715839
137892645
598436271
423178596
816549723
759623418
375281964
982364157
641957382
"
        );
    }

    #[test]
    fn arto_inkala() {
        let sudoku: Sudoku = "\
8........
..36.....
.7..9.2..
.5...7...
....457..
...1...3.
..1....68
..85...1.
.9....4..
"
        .parse()
        .unwrap();
        assert_eq!(
            sudoku.solve().unwrap().to_string(),
            "\
8........
..36.....
.7..9.2..
.5...7...
....457..
...1...3.
..1....68
..85...1.
.9....4..
"
        )
    }
}
