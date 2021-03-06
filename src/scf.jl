
import PySCF: pyscf, pyscf_atom_from_xyz, get_4idx
import PySCF: diis

abstract type HartreeFockModel end

struct RestrictedHartreeFock <: HartreeFockModel
   Emol
   e
   C
   #S
   #D
end

function buildFock(hcore,D,nao,eri)
   F = copy(hcore)
   for i in 1:nao, j in 1:nao, # add Gμν*P
       k in 1:nao, l in 1:nao
      ijkl = get_4idx(i,j,k,l)
      ikjl = get_4idx(i,k,j,l)
      F[i,j] += D[k,l]*(2eri[ijkl]-eri[ikjl])
   end
   F
end

# Löwdin symmetric Orthogonalization (with A = X^(-1/2))
function lowdinOrtho(A,B)
   A' * B * A
end

#########################################################################
# Project #3: The Hartree-Fock self-consistent field (SCF) procedure.
#########################################################################
function scf_rhf(mol; verbose=true)
   nelec = mol.nelectron
   nocc  = mol.nelectron ÷ 2 
   nao   = mol.nao_nr() |> n -> convert(Int,n) # why not converted ??!
   enuc  = mol.energy_nuc()
   s     = mol.intor("cint1e_ovlp_sph")
   hcore = mol.intor("cint1e_nuc_sph") + mol.intor("cint1e_kin_sph")
   eri   = mol.intor("cint2e_sph", aosym="s8")

   ########################################################
   #4: BUILD THE ORTHOGONALIZATION MATRIX
   ########################################################
   # NOTE: The AO basis is non-orthogonal

   # Symmetric orthogonalization matrix
   s_minushalf = s^(-0.5) # Matrix power (via squaring)

   ########################################################
   #5: BUILD THE INITIAL (GUESS) DENSITY
   ########################################################

   P = zeros(nao,nao) # D = 2P, P often called density matrix

   ########################################################
   #6: COMPUTE THE INITIAL SCF ENERGY
   ########################################################

   Eelec = 0.0

   #*******************************************************
   #*******************************************************
   #
   #             SCF ITERATIVE PROCESS
   #
   #*******************************************************
   #*******************************************************

   iteri     = 0    # extant iteration
   itermax   = 100  # max number of iterations
   δE        = 1e-8 # energy threshold for convergence
   δP        = 1e-8 # P-matrix threshold for convergence
   ΔE        = 1.0; @assert ΔE > δE # initial E difference
   ΔP        = 1.0; @assert ΔP > δP # initial P difference
   Eelec     = 0.0
   oldEelec  = 0.0
   Emol      = 0.0  # total molecular energy
   e         = zeros(nao)
   C         = zeros(nao,nao)
   DIISstack = [] # for DIIS
   Fstack    = [] # for DIIS
   if verbose
      println(" Iter        E(elec)        E(tot)        ΔE(elec)         ΔP")
      println("-----   --------------  -------------  ------------  -------------")
   end

   while abs(ΔE) > δE && iteri < itermax && ΔP > δP

      iteri += 1

      ########################################################
      #7: COMPUTE THE NEW FOCK MATRIX
      ########################################################

      F = buildFock(hcore,P,nao,eri)

      F, Fstack, DIISstack = diis(F, Fstack, DIISstack, s_minushalf, s, P, iteri)

      ########################################################
      #8: BUILD THE NEW DENSITY MATRIX
      ########################################################

      # Transformed Fock matrix
      Fp = lowdinOrtho(s_minushalf,F)
      # MO coeffs as linear combination of orthonormal MO basis functions
      e, Cp = eigen(Hermitian(Fp))
      # New MO Coefficients as linear combination of AO basis functions
      C = s_minushalf * Cp
      # Renew P-matrix.
      P, oldP = zeros(nao,nao), copy(P)

      mocc = C[:,1:nocc]
      P  = mocc * mocc'
      ΔP = norm(P - oldP) # Frobenius norm

      ########################################################
      #9: COMPUTE THE NEW SCF ENERGY
      ########################################################

      Eelec, oldEelec = tr(P * (hcore + F)), Eelec # eq. (238) Janos

      ΔE   = Eelec - oldEelec
      Emol = Eelec + enuc

      if verbose
         printfmt(" {1:3d}  {2:14.8f}  {3:14.8f}  {4:14.8f} {5:14.8f}\n",
                    iteri,   Eelec,       Emol,      ΔE   ,    ΔP)
      end
      ########################################################
      #10: TEST FOR CONVERGENCE: WHILE LOOP (go up)
      ########################################################
   end
   @assert iteri < itermax "NOT CONVERGED!!"

#   covalent_bond_order(P,s,mol)
#
#   rdm1 = fock_dirac_denmat(C,s,2P)
#   nelec_rdm1 = tr(rdm1)
#   @assert isapprox(nelec, nelec_rdm1, atol=1e-8)
#   printfmt("Population: {:.3f} elec \n", nelec_rdm1)
#
#   rdm2 = second_denmat(rdm1)
#   # TODO: energy from density matrices eqs. (224,227) Janos

   RestrictedHartreeFock(Emol,e,C)
end

