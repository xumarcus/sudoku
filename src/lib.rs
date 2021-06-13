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
            for cell in core::RowIterator::new(*row) {
                if cell.next_power_of_two() == cell {
                    write!(f, "{}", (cell.trailing_zeros() as u8) + b'1')?;
                } else {
                    write!(f, ".")?;
                }
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

struct Backtrack {
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

enum Solutions {
    Multiple(std::iter::Flatten<Backtrack>),
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
        self.0.iter().cloned().flat_map(core::RowIterator::new)
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
                Solutions::Multiple(Backtrack::new(self, idx, bits).flatten())
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
            let domain = buf_ & mask;
            core::BitIterator::new(domain)
                .all(|bit| {
                    if matchings & bit == 0 {
                        let pruned = (buf_ & !mask) | bit;
                        if let Some(matching) = core::kuhn(pruned) {
                            matchings |= matching;
                            return false;
                        } else {
                            buf_ &= !bit;
                            return true;
                        }
                    }
                })
                .then(|| ())?;
        }
        debug_assert_eq!(buf, matchings);
        Some(
            core::BitIterator::new(indices)
                .zip(core::RowIterator::new(buf).zip(core::RowIterator::new(buf_)))
                .filter_map(|(bit, (row, row_))| (row != row_).then(|| bit))
                .sum::<u128>(),
        )
    }
}

mod core {
    use super::*;

    pub struct RowIterator {
        st: u128,
    }

    impl RowIterator {
        pub fn new(st: u128) -> Self {
            Self { st }
        }
    }

    impl Iterator for RowIterator {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            (self.st != 0).then(|| {
                let item = self.st & S09;
                self.st >>= N2;
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
}

/*
mod core {
    use super::*;

    pub fn enforce(
        data: &mut [Bits; N4],
        indices: &[usize; N2],
        queue: &mut VecDeque<usize>,
    ) -> Option<()> {
        let u_0 = from_indices(data, indices);
        let mut u_1 = [0usize; N2];
        for (i, z) in u_0.iter().enumerate() {
            for k in 0..N2 {
                let bit = 1 << k;
                if u_1[i] & bit == 0 && z & bit != 0 {
                    let mut u_p = u_0.clone();
                    u_p[i] = bit;
                    if let Some(x_of) = crate::core::kuhn(&u_p) {
                        for (y, x) in x_of.iter().enumerate() {
                            u_1[*x] |= 1 << y;
                        }
                    }
                }
            }
        }
        u_1.iter().all(|bits| *bits != 0).then(|| {
            for (i, (u_0_i, u_1_i)) in u_0.iter().zip(u_1.iter()).enumerate() {
                if u_0_i != u_1_i {
                    let idx = indices[i];
                    data[idx] = *u_1_i;
                    queue.push_back(idx);
                }
            }
            Some(())
        })?
    }

    // No generic parameter: RFC 2000 #44580
    fn from_indices<T: Copy + Default>(buf: &[T], indices: &[usize; N2]) -> [T; N2] {
        let mut new_buf = [T::default(); N2];
        for (i, idx) in indices.iter().enumerate() {
            new_buf[i] = buf[*idx];
        }
        new_buf
    }

    fn dfs(u: &[Bits; N2], idx: usize, x_of: &mut [usize; N2], mut vis: Bits) -> bool {
        vis |= 1 << idx;
        let u_i = u[idx];
        for i in 0..N2 {
            if u_i & (1 << i) != 0 {
                let x = x_of[i];
                if x == N2 || (vis & (1 << x) == 0 && dfs(u, x, x_of, vis)) {
                    x_of[i] = idx;
                    return true;
                }
            }
        }
        false
    }

    fn kuhn(u: &[Bits; N2]) -> Option<[usize; N2]> {
        let mut x_of = [N2; N2];
        dfs(&u, 0, &mut x_of, 0);
        dfs(&u, 1, &mut x_of, 0);
        dfs(&u, 2, &mut x_of, 0);
        dfs(&u, 3, &mut x_of, 0);
        dfs(&u, 4, &mut x_of, 0);
        dfs(&u, 5, &mut x_of, 0);
        dfs(&u, 6, &mut x_of, 0);
        dfs(&u, 7, &mut x_of, 0);
        dfs(&u, 8, &mut x_of, 0);
        Some(x_of).filter(|x| x.iter().all(|z| *z != N2))
    }

    #[allow(dead_code)]
    fn y_of(x_of: &[usize; N2]) -> [usize; N2] {
        let mut y_of = [0usize; N2];
        for (i, z) in x_of.iter().enumerate() {
            y_of[*z] = i;
        }
        y_of
    }
}
*/

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
            sudoku.to_string(),
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
            sudoku.to_string(),
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
            sudoku.to_string(),
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
            sudoku.to_string(),
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
