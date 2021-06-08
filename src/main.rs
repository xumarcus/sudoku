use std::collections::VecDeque;
use std::env;
use std::fmt::{self, Display};
use std::io::{self, Read};
use std::str::FromStr;

use strum::EnumCount;
use thiserror::Error;

#[derive(strum_macros::EnumCount)]
enum ConstraintType {
    Block,
    Row,
    Col,
}

type Bits = usize;
type Arcs = [[[usize; N2]; ConstraintType::COUNT]; N4];

const N1: usize = 3;
const N2: usize = N1 * N1;
const N4: usize = N2 * N2;
const ALL_BITS: Bits = (1 << N2) - 1;
const ARCS: Arcs = crate::core::arcs_init();

#[derive(Clone, Debug, Error)]
#[error("No solution")]
struct NoSolutionError;

#[derive(Clone, Debug)]
struct Sudoku([usize; N4]);

impl FromStr for Sudoku {
    type Err = NoSolutionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let iter = s.chars().filter(char::is_ascii_digit).map(|d| d as u8);
        Sudoku::new(iter).ok_or(NoSolutionError)
    }
}

impl Display for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in self.0.chunks_exact(N2) {
            let s = chunk
                .iter()
                .map(|mask| match mask {
                    1 => '1',
                    2 => '2',
                    4 => '3',
                    8 => '4',
                    16 => '5',
                    32 => '6',
                    64 => '7',
                    128 => '8',
                    256 => '9',
                    _ => '.',
                })
                .collect::<String>();
            writeln!(f, "{}", s)?;
        }
        Ok(())
    }
}

impl Sudoku {
    fn set(&mut self, idx: usize, digit: usize) -> Option<()> {
        let bit = 1 << digit;
        (self.0[idx] & bit != 0).then(|| {
            self.0[idx] = bit;
            let mut queue = std::iter::once(idx).collect::<VecDeque<usize>>();
            while let Some(cur) = queue.pop_front() {
                for indices in ARCS[cur].iter() {
                    let u = crate::core::from_indices(&self.0, indices);
                    crate::core::enforce(&u, indices, &mut self.0, &mut queue)?;
                }
            }
            Some(())
        })?
    }

    pub fn new(it: impl Iterator<Item = u8>) -> Option<Self> {
        let mut sudoku = Sudoku([ALL_BITS; N4]);
        for (idx, chr) in it.enumerate() {
            if chr != b'0' {
                sudoku.set(idx, (chr - b'1') as usize)?;
            }
        }
        Some(sudoku)
    }

    pub fn backtrack(self) -> Option<Sudoku> {
        for (idx, val) in self.0.iter().enumerate() {
            if *val == 0 {
                return None;
            }
            if val.next_power_of_two() != *val {
                return (0..N2)
                    .filter_map(|i| {
                        (val & (1 << i) != 0).then(|| {
                            let mut clone = self.clone();
                            clone.set(idx, i)?;
                            clone.backtrack()
                        })?
                    })
                    .next();
            }
        }
        Some(self)
    }
}

mod core {
    use crate::*;

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

    // No generic parameter: RFC 2000 #44580
    pub fn from_indices<T: Copy + Default>(buf: &[T], indices: &[usize; N2]) -> [T; N2] {
        // Skip frequent bound-checking
        /* unsafe {
            let mut new_buf: [T; N2] = std::mem::zeroed();
            for (i, idx) in indices.iter().enumerate() {
                *new_buf.get_unchecked_mut(i) = *buf.get_unchecked(*idx);
            }
            new_buf
        } */
        let mut new_buf = [T::default(); N2];
        for (i, idx) in indices.iter().enumerate() {
            new_buf[i] = buf[*idx];
        }
        new_buf
    }

    pub fn enforce(
        u: &[Bits; N2],
        indices: &[usize; N2],
        data: &mut [Bits; N4],
        queue: &mut VecDeque<usize>,
    ) -> Option<()> {
        let mut u_1 = [0usize; N2];
        for (i, z) in u.iter().enumerate() {
            for k in 0..N2 {
                let bit = 1 << k;
                if u_1[i] & bit == 0 && z & bit != 0 {
                    let mut u_p = u.clone();
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
            for (i, (u_i, u_1_i)) in u.iter().zip(u_1.iter()).enumerate() {
                if u_i != u_1_i {
                    let idx = indices[i];
                    data[idx] = *u_1_i;
                    queue.push_back(idx);
                }
            }
            Some(())
        })?
    }

    fn dfs(u: &[Bits; N2], idx: usize, x_of: &mut [usize; N2], mut vis: Bits) -> bool {
        vis |= 1 << idx;
        let u_i = u[idx]; /* unsafe { *u.get_unchecked(idx) }; */
        for i in 0..N2 {
            /* unsafe { *x_of.get_unchecked(i) } */
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
            /* unsafe {
                *y_of.get_unchecked_mut(*z) = i;
            } */
            y_of[*z] = i;
        }
        y_of
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sudoku = match env::args().skip(1).next() {
        Some(s) => s,
        None => {
            let mut stdin = io::stdin();
            let mut buf = String::new();
            println!("Started in interactive mode. Enter EOF (Ctrl + Z) once done.");
            stdin.read_to_string(&mut buf)?;
            buf
        }
    }
    .parse::<Sudoku>()?;
    if let Some(solved) = sudoku.backtrack() {
        println!("{}", solved);
    }
    Ok(())
}
