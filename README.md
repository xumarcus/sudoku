GAC-based Sudoku Solver
=======================
Sudoku can be modelled as a constraints satisfaction problem. This solver 
eliminates candidates using AllDifferent GAC inconsistencies with a naive
implementation of Kuhn algorithm for maximum bipartite matching.

# Benchmarks (R5 2600)
- v0.1 on CodeGolf @ StackOverflow: ~750s

# Further optimizations
- Run pre-processing before enforcing GAC
- Do not enforce twice on same constraints
- Iterate on LSBs wherever possible
- SIMD/AVX512 Config
- Profiling

# TODO
- Convert to U128
- Separate I/O & formatting