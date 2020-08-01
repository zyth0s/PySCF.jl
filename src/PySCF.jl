
module PySCF

using PyCall: pyimport

pyscf = pyimport("pyscf")
mp = pyimport("pyscf.mp")   # Had to import mp alone ??!
cc = pyimport("pyscf.cc") # Had to import mp alone ??!
np = pyimport("numpy") # alternative: TensorOperations.jl, Einsum.jl,

# Utilities
function pyscf_atom_from_xyz(fpath::String)
   join(split(read(open(fpath),String),"\n")[3:end],"\n")
end

function index(i::Int,j::Int)
   m,M = minmax(i-1,j-1)
   M*(M+1)÷2 + m + 1
end

# Compound indices ijkl
function get_4idx(i::Int,j::Int,k::Int,l::Int)
   ij = index(i,j)
   kl = index(k,l)
   index(ij,kl)
end

# Calculation of the electronic structure of a molecule
# with the help of PySCF to calculate AO integrals and hold molecular data
# * Restricted Hartree-Fock
# * Møller-Plesset order 2
# * Coupled Cluster Singles and Doubles
# Calculates the energy of any molecule.
# Instructions at https://github.com/CrawfordGroup/ProgrammingProjects
# may be of interest.

using Formatting: printfmt
using LinearAlgebra: Diagonal, Hermitian, eigen, norm, tr, diag, dot

include("scf.jl")
include("mp2.jl")
include("ccsd.jl")

end
