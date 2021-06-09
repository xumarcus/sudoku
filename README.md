// Copyright (C) 2021 MarcusXu
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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