use std::collections::VecDeque;
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
const N4: usize = N2 * N2;
const S9: u128 = 0x111111111;
const ROW_HEAD: u128 = S9 << (N4 - N2);

#[derive(Clone, Debug)]
pub struct Sudoku([u128; N2]);

impl FromStr for Sudoku {
    type Err = ParseSudokuError;

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
                let sudoku: [u128; N2] = buf
                    .chunks_exact(N2)
                    .map(|chunk| chunk.iter().fold(0u128, |acc, x| (acc << N2) + x))
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
        for mut row in self.0.iter().cloned() {
            for _ in 0..N2 {
                write!(f, "{}", (row & ROW_HEAD).count_ones())?;
                row >>= N2;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Sudoku {
    pub fn solve(&self) -> Option<Self> {
        self.solutions().next()
    }

    pub fn solutions(&self) -> SudokuSolutions {
        SudokuSolutions(self.clone())
    }

    pub fn generate() -> Self {
        unimplemented!()
    }
}

#[derive(Clone, Debug)]
struct SudokuSolutions(Sudoku);

impl Iterator for SudokuSolutions {
    type Item = Sudoku;

    fn next(&mut self) -> Option<Self::Item> {

    }
}

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
438526917
796318452
"
       