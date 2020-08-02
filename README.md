[![Build Status](https://travis-ci.com/zyth0s/PySCF.jl.svg?branch=master)](https://travis-ci.com/zyth0s/PySCF.jl)
[![codecov](https://codecov.io/gh/zyth0s/PySCF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/zyth0s/PySCF.jl)

# PySCF.jl
Julia interface to PySCF via PyCall

Primarily, PySCF is used as an aid and to check correctness of our own implementations:

- [x] SCF HF energy
- [x] DIIS
- Properties:
  - [x] Dipole moment
  - [x] Generalized AO population analysis (e.g. Mulliken, Löwdin, ...)
  - [x] Number of electron pairs (J, XC, Total)
- [x] MP2 energy
- [x] CCSD energy
