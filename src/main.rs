use std::array::IntoIter;
use std::env;
use std::io;
use std::io::Read;
use std::str::FromStr;

use thiserror::Error;

const N1: usize = 3;
const N2: usize = N1 * N1;
const N4: usize = N2 * N2;

#[derive(Clone, Debug, Error)]
#[error("81 digits expected")]
struct SudokuError;

#[derive(Clone, Debug)]
struct Sudoku([usize; N4]);

type Bits = usize;

impl FromStr for Sudoku {
    type Err = SudokuError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Sudoku::try_collect(
            s.chars()
                .filter(char::is_ascii_digit)
                .take(N4)
                .map(|d| d as u8),
        )
    }
}

impl Sudoku {
    #[inline]
    fn set(&mut self, idx: usize, digit: usize) -> Result<(), SudokuError> {
        
    }

    fn try_collect(it: impl Iterator<Item = u8>) -> Result<Self, SudokuError> {
        let mut sudoku = Sudoku([(1 << N2) - 1; N4]);
        for (idx, digit) in it.enumerate() {
            if digit != b'0' {
                sudoku[idx] = 1 << (digit - b'0');
                sudoku.enforce_GAC(idx)?;
            }
        }
        Ok(sudoku)
    }
}

fn dfs(u: &[Bits; N2], idx: usize, x_of: &mut [usize; N2], mut vis: Bits) -> bool {
    vis |= 1 << idx;
    let u_i = unsafe { *u.get_unchecked(idx) };
    for i in 0..N2 {
        if u_i & (1 << i) == 0 {
            continue;
        }
        if unsafe { *x_of.get_unchecked(i) } != N2 {
            if vis & (1 << i) != 0 || !dfs(u, i, x_of, vis) {
                continue;
            }
        }
        unsafe {
            *x_of.get_unchecked_mut(i) = idx;
        }
        return true;
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
    x_of.iter().all(|z| *z != N2).then(|| {
        let mut y_of = [0usize; N2];
        for (i, z) in x_of.iter().enumerate() {
            unsafe {
                *y_of.get_unchecked_mut(*z) = i;
            }
        }
        y_of
    })
}

// In theory with GAC BT should be less often
fn solve(sudoku: Sudoku) -> Option<Sudoku> {
    sudoku
        .0
        .iter()
        .enumerate()
        .find(|(_, val)| val.next_power_of_two() != **val)
        .and_then(|(idx, val)| {
            (0..N2)
                .filter_map(|i| {
                    // Workaround
                    (val & (1 << i) != 0).then(|| ()).and_then(|_| {
                        let mut cl = sudoku.clone();
                        cl.0[idx] = 1 << i;
                        enforce_GAC(&mut cl, idx);
                        solve(cl)
                    })
                })
                .next()
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sudoku = match env::args().skip(1).next() {
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
    for (idx, bits) in IntoIter::new(sudoku.0).enumerate() {
        gac_enforce(&mut sudoku, idx);
    }
    Ok(())
}
