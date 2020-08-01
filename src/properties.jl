
import PySCF: pyimport

#########################################################################
# Additional Concepts
# also see: "Hartree-Fock methods of Quantum Chemistry" by Janos Angyan
#########################################################################
# One-electron properties
function dipole_moment(D, mol)
   # eq. (365) Janos
   # It does the same as PySCF does.
   #ao_dip = @pywith mol.with_common_orig([0,0,0]) begin
   #   mol.intor_symmetric("int1e_r", comp=3)
   #end
   common_orig_bak = mol._env[1:3]
   mol = mol.set_common_orig([0,0,0])
   ao_dip = mol.intor_symmetric("int1e_r", comp=3)
   mol = mol.set_common_orig(common_orig_bak)

   np = pyimport("numpy")
   el_dip   = real.(np.einsum("xij,ji->x", ao_dip, D))
   charges  = mol.atom_charges()
   coords   = mol.atom_coords()
   nucl_dip = np.einsum("i,ix->x", charges, coords)
   mol_dip  = nucl_dip - el_dip
   pyscf = pyimport("pyscf")
   mol_dip *= pyscf.data.nist.AU2DEBYE
end

# Population Analysis/Atomic Charges
@doc raw"""
   `population_analysis(D,S,α,mol)`

Generalized Mulliken, Lowdin, ... population analysis
with `D` the AO density matrix, `S` the overlap matrix,
`α` defines the analysis type, and mol is a PySCF molecule.

Qa = Za - ∑μ∈a [S^α P S^(1-α)]μμ  with 0 < α < 1.

In particular:
* α = 1 => Mulliken doi:10.1063/1.1740588
* α = ½ => Löwdin
"""
function population_analysis(D,S,α,mol)
   s_α, s_mα = S^α, S^(1-α) # Matrix powers (via squaring)
   @assert isapprox(mol.nelectron, tr(s_α*D*s_mα), atol=1e-8) # eq. (363) Janos
   PopAO = diag(s_α * D * s_mα)
   Q = zeros(mol.natm)
   for (μ,record) in enumerate(mol.ao_labels(fmt=nothing))
      a = record[1] + 1 |> n -> convert(Int,n)
      Q[a] -= PopAO[μ] # eqs. (362,364) Janos
   end
   Q += mol.atom_charges()
   for a in eachindex(Q)
      println(" Atom $a has charge $(Q[a])")
   end
   @assert isapprox(sum(Q), mol.charge, atol=1e-8)
end

function covalent_bond_order(P,S,mol)
   natm = mol.natm
   nelec = mol.nelectron
   @show nelec
   @show Npairs = sum( (2P * S) * (2P * S) ) # RHF
   ao_atom = first.(mol.ao_labels(fmt=nothing))
   B_ab = zeros(natm,natm)
   for a in 1:natm, b in a:natm
      inds = findall( x -> isequal(x+1,a) || isequal(x+1,b), ao_atom)
      #@show inds
      #Pab = reshape(P[inds,inds],length(inds),length(inds))
      #Sab = reshape(S[inds,inds],length(inds),length(inds))
      Pab = P[inds,inds]
      Sab = S[inds,inds]
      # eq. (375) Janos
      #Bab = sum( (Pab * Sab)*(Pab * Sab)' + (Pab * Sab)*(Pab * Sab)' ) # RHF
      Bab = 2dot( Pab * Sab, Pab * Sab) # RHF
      println("$a  $b  $Bab")
   end
end

# Hartree-Fock Reduced Density Matrices
@doc raw"""
   fock_dirac_denmat(C,S,D)

1RDM in MO basis. For Hartree-Fock it is idempotent, diagonal with 1 or 0 elements.
Therefore it is idempotent. And it is called Fock-Dirac density matrix.

`C` is the matrix of cannonical coefficients, `S` is the overlap matrix
and `D` is the density matrix in the basis of AO.
 """
function fock_dirac_denmat(C,S,D)
   # C' S D S C
   #np.einsum("ji,jk,kl,lm,mn->in",C,S,D,S,C,optimize=true)
   C' * S * D * S * C
end

"""
2RD
"""
function second_reduced_density(rdm1)
   #rdm2D = zeros(nao,nao)
   #for i in 1:nao, j in 1:nao
   rdm2D = similar(rdm1)
   for (i,j) in Tuple.(CartesianIndices(rdm1))
      rdm2D[i,j] = rdm1[i,i]*rdm1[j,j] - rdm1[i,j]*rdm1[j,i]
   end
   rdm2D
end

"""
2RDM
"""
function second_denmat(D)
   # Clearly explained in sections 11.15, 13.3 of Janos Angyan notes
   #rdm2 = zeros(nsmo,nsmo,nsmo,nsmo)
   #for i in 1:nao, j in 1:nao, k in 1:nao, l in 1:nao
   rdm2 = zeros(size(D)...,size(D)...)
   for (i,j,k,l) in Tuple.(CartesianIndices(rdm2))
      # 1st RHS is the Coulomb component ∫ = N^2
      # 2nd RHS is the exchange component ∫ = N
      #rdm2[l,k,i,j] =     D[l,i] * D[k,j] - D[l,j] * D[k,i] # eq. (338) Janos
      rdm2[i,j,k,l] = 0.5(D[i,j] * D[k,l] - D[i,l] * D[k,j]) # eq. (6) J. of Math. Chem. 28, Nos. 1–3, 2000
   end
   #for i in 1:nao,j in 1:nao,k in 1:nao,l in 1:nao
      #lk, ij = index(l,k), index(i,j)
      #rdm2[l,k,i,j] = D[l,i]*D[j,k] - D[l,j]*D[k,i]
      #rdm2[l,j,i,k] = D[l,i]*D[k,j] - D[l,k]*D[j,i]
   #end
   rdm2
end

