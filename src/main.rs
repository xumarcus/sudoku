use sudoku::Sudoku;

use std::env;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sudoku = match env::args().skip(1).next() {
        Some(s) => s,
        None => {
            let mut stdin = io::stdin();
            let mut buf = String::new();
            println!(
                "\
Started in interactive mode.
`.` for blanks.
Enter EOF (Ctrl + Z) once done.
"
            );
            stdin.read_to_string(&mut buf)?;
            buf
        }
    }
    .parse::<Sudoku>()?;
    if let Some(solved) = sudoku.solve() {
        println!("{}", solved);
    }
    Ok(())
}
