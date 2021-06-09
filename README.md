GAC-based Sudoku Solver
=======================
Sudoku can be modelled as a constraints satisfaction problem. This solver 
eliminates candidates using AllDifferent GAC inconsistencies with a naive
implementation of Kuhn algorithm for maximum bipartite matching.

# Benchmarks (R5 2600)
- v0.1 on CodeGolf @ StackOverflow: ~750s

# TODO
- Run pre-processing before enforcing GAC
- Do not enforce twice on same constraints
- Create new SudokuSolver struct and separate I/O | formatting
- Iterate on LSBs wherever possible
- SIMD/AVX512 Config/optimizations
- Profiling