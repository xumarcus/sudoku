use std::cmp::Ordering;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::fmt::{self, Display};
use std::str::{self, FromStr};

use itertools::unfold;
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
const N4: usize = N2 * N2;
const S9: u128 = 0x111111111;

// Minimize interdependence
const INITIAL_CONSTRAINT_INDICES: [usize; N2] = [0, 40, 80, 28, 68, 24, 56, 12, 52];

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
            for cell in core::iter_row(*row) {
                let byte = match core::log2(cell) {
                    Some(x) => (x as u8) + b'1',
                    None => b'.',
                };
                write!(f, "{}", byte)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Sudoku {
    pub fn solve(self) -> Option<Self> {
        self.solutions().next()
    }

    pub fn solutions(mut self) -> Box<dyn Iterator<Item = Sudoku>> {
        let queue = INITIAL_CONSTRAINT_INDICES.iter().cloned().collect();
        if !self.make_consistent(queue) {
            Box::new(std::iter::empty())
        } else {
            self.solutions_from_consistent()
        }
    }

    pub fn generate() -> Self {
        unimplemented!()
    }

    fn iter(&self) -> impl Iterator<Item = u128> + '_ {
        self.0.iter().cloned().flat_map(core::iter_row)
    }

    // Boxing to prevent recursive opaque type
    fn solutions_from_consistent(self) -> Box<dyn Iterator<Item = Sudoku>> {
        if let Some((idx, bits, _)) = self
            .iter()
            .enumerate()
            .filter_map(|(idx, bits)| {
                let cnt = bits.count_ones();
                (cnt > 1).then(|| (idx, bits, cnt))
            })
            .min_by_key(|t| t.2)
        {
            // If guesses are minimal, just clone
            Box::new(
                core::iter_bits(bits)
                    .filter_map(move |bit| {
                        let mut cp = self.clone();
                        let shl = N2 * (idx % N2);
                        cp.0[idx / N2] &= (S9 << shl).wrapping_neg() | (bit << shl);
                        let queue = std::iter::once(idx).collect();
                        cp.make_consistent(queue)
                            .then(|| cp.solutions_from_consistent())
                    })
                    .flatten(),
            )
        } else {
            Box::new(std::iter::once(self))
        }
    }

    fn make_consistent(&mut self, queue: VecDeque<usize>) -> bool {
        while let Some(idx) = queue.pop_front() {
            // iterate self & (S9 << N2 * idx % N2) -> u128)
        }
    }
}

mod core {
    use super::*;
    pub fn iter_row(row: u128) -> impl Iterator<Item = u128> {
        unfold(row, |st| {
            // Special case?
            (*st != 0).then(|| {
                let item = *st & S9;
                *st >>= N2;
                item
            })
        })
    }

    pub fn iter_bits(bits: u128) -> impl Iterator<Item = u128> {
        unfold(bits, |st| {
            (*st != 0).then(|| {
                let item = *st & st.wrapping_neg();
                *st -= item;
                item
            })
        })
    }

    pub fn log2(cell: u128) -> Option<usize> {
        match cell {
            0x1 => Some(0),
            0x10 => Some(1),
            0x100 => Some(2),
            0x1000 => Some(3),
            0x10000 => Some(4),
            0x100000 => Some(5),
            0x1000000 => Some(6),
            0x10000000 => Some(7),
            0x100000000 => Some(8),
            _ => None,
        }
    }
}

/*
mod core {
    use super::*;

    pub const fn arcs_init() -> Arcs {
        let mut arcs = [[[0; N2]; N1]; N4];
        let mut blocks = [[0; N2]; N2];
        let mut i = 0;
        while i < N1 {
            let mut j = 0;
            while j < N1 {
                let mut k = 0;
                while k < N1 {
                    let mut l = 0;
                    while l < N1 {
                        blocks[i * N1 + j][k * N1 + l] = (i * N2 + j) * N1 + k * N2 + l;
                        l += 1;
                    }
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < N2 {
            let mut j = 0;
            while j < N2 {
                arcs[i * N2 + j][ConstraintType::Block as usize] = blocks[(i % N1) * N1 + (j % N1)];
                let mut k = 0;
                while k < N2 {
                    arcs[i * N2 + j][ConstraintType::Row as usize][k] = i * N2 + k;
                    arcs[i * N2 + j][ConstraintType::Col as usize][k] = k * N2 + j;
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
        arcs
    }

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
        );
    }
}
/*
        assert_eq!(
            sudoku.backtrack().unwrap().to_string(),
            "\
812753649
943682175
675491283
154237896
369845721
287169534
521974368
"
        );
*/
