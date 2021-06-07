use std::convert::TryInto;
use std::array::IntoIter;
use std::env;
use std::io;
use std::io::Read;
use std::str;

use thiserror::Error;

const N1: usize = 3;
const N2: usize = N1 * N1;
const N4: usize = N2 * N2;

#[derive(Clone, Debug, Error)]
#[error("81 digits expected")]
struct ParseSudokuError;

#[derive(Clone, Debug)]
struct Sudoku([usize; N4]);

type Bits = usize;

impl str::FromStr for Sudoku {
    type Err = ParseSudokuError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.chars()
        .filter(char::is_ascii_digit)
        .take(N4)
        .map(|d| match d as u8 - b'0' {
            0 => 0x111111111,
            x => 1 << x,
        })
        .collect::<Vec<Bits>>()
        .try_into()
        .map(Sudoku)
        .map_err(|_| ParseSudokuError)
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
        unsafe { *x_of.get_unchecked_mut(i) = idx; }
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
    if x_of.iter().all(|z| *z != N2) {
        let mut y_of = [0usize; N2];
        for (i, z) in x_of.iter().enumerate() {
            unsafe { *y_of.get_unchecked_mut(*z) = i; }
        }
        Some(y_of)
    } else {
        None
    }
}

fn gac_enforce(sudoku: &mut Sudoku, idx: usize) -> Vec<usize> {
    vec![]
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
    }.parse::<Sudoku>()?;
    for (idx, bits) in IntoIter::new(sudoku.0).enumerate() {
        gac_enforce(&mut sudoku, idx);
    }
    Ok(())
}