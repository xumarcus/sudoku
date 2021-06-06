use std::convert::TryInto;
use std::env;
use std::io;
use std::io::Read;
use std::str;

use thiserror::Error;

const SUDOKU_SIZE: usize = 81;

#[derive(Clone, Debug, Error)]
#[error("81 digits expected")]
struct ParseSudokuError;

#[derive(Clone, Debug)]
struct Sudoku([usize; SUDOKU_SIZE]);

type Bits = usize;

impl str::FromStr for Sudoku {
    type Err = ParseSudokuError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.chars()
        .filter(char::is_ascii_digit)
        .take(SUDOKU_SIZE)
        .map(|d| match d as u8 - b'0' {
            0 => 0x111111111,
            x => (1 << x) - 1,
        })
        .collect::<Vec<Bits>>()
        .try_into()
        .map(Sudoku)
        .map_err(|_| ParseSudokuError)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sudoku = match env::args().next() {
        Some(s) => s,
        None => {
            let stdin = io::stdin();
            let mut buf = String::new();
            println!("Started in interactive mode. Enter EOF (Ctrl + D) once done.");
            stdin.read_to_string(&mut buf)?;
            buf
        }
    }.parse::<Sudoku>()?;
    Ok(())
}
 