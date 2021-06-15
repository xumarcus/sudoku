use sudoku::Sudoku;

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .skip(1)
        .next()
        .unwrap_or("codegolf.in".to_string());
    let stdin = io::BufReader::new(fs::File::open(path)?);
    let mut lines = stdin.lines();
    let n = std::cmp::min(
        5000,
        match lines.next() {
            None => 0,
            Some(s) => s?.parse::<usize>()?,
        },
    );

    let mut stdout = io::BufWriter::new(fs::File::create("codegolf.out".to_string())?);
    writeln!(stdout, "{}", n)?;
    for line in lines.take(n) {
        let line = line?;
        let sudoku = line.parse::<Sudoku>()?;
        if let Some(solution) = sudoku.solve() {
            let sol = solution
                .to_string()
                .chars()
                .filter(|x| *x != '\n')
                .collect::<String>();
            writeln!(stdout, "{},{}", line, sol)?;
        }
    }
    Ok(())
}
