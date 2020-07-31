
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Project #5: The Coupled Cluster Singles and Doubles (CCSD) Energy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# J. Chem. Phys. 94, 4334, doi:10.1063/1.460620

# We follow the convention that
# i, j, k,... represent occupied orbitals, with
# a, b, c,... unoccupied.
# p, q, r,... are generic indices which may represent
#             either occupied or unoccupied
# orbitals.

###################################################
#1: Preparing the Spin-Orbital Basis Integrals
###################################################
# Translate integrals from spatial mo basis to spin-orbital basis
function orb_to_spinorb(e::Array{T,1},     # eigE of spatial-MO orbitals
                        eri_mo::Array{T,1} # ERIs in spatial-MO basis
                       ) where T <: AbstractFloat
   # 1  2  3  4  5  6  7  8       (spin-orbital no.)
   # α, β, α, β, α, β, α, β, ...  (spin component
   # χ₁,χ₁,χ₂,χ₂,χ₃,χ₃,χ₄,χ₄, ... (spatial component)
   # In RHF closed shell both α and β have the same energy: map ϵ₁ ↦ ϵ₁,ϵ₂; ϵ₂ ↦ ϵ₃,ϵ₄; ...
   # cld(1,2) and cld(2,2) → 1, cld(3,2) and cld(4,2) → 2, ...
   nao  = length(e)
   nsmo = 2nao
   eri_smo = zeros(nsmo,nsmo,nsmo,nsmo)
   @inbounds (
   for p in 1:nsmo, q in 1:nsmo, r in 1:nsmo, s in 1:nsmo
      prqs = get_4idx( cld.((p,r,q,s), 2)...) # ceil division cld(p,2) == (p+1)÷2 with p ≥ 0
      psqr = get_4idx( cld.((p,s,q,r), 2)...) # broadcast and unpack arguments
      value1 = eri_mo[prqs] * (iseven(p) == iseven(r)) * (iseven(q) == iseven(s))
      value2 = eri_mo[psqr] * (iseven(p) == iseven(s)) * (iseven(q) == iseven(r))
      eri_smo[p,q,r,s] = value1 - value2
   end )
   # The fock matrix is diagonal in the spin-MO basis
   fs = Diagonal(zeros(nsmo))
   for i in 1:nsmo
      fs.diag[i] = e[cld(i,2)]
   end
   eri_smo, fs
end

###################################################
#2: Build the Initial-Guess Cluster Amplitudes
###################################################

function initial_amplitudes(occupied,virtual,
                            fs::Diagonal{T,Array{T,1}}, # Fock matrix in spin-MO basis
                            eri_smo::Array{T,4}         # ERIs        in spin-MO basis
                            ) where T <: AbstractFloat
   nsmo = length(occupied) + length(virtual)
   t1 = zeros(nsmo,nsmo)
   t2 = zeros(nsmo,nsmo,nsmo,nsmo)

   # MP₂ guess for T₂ amplitudes
   # t[a,b,i,j] = ⟨ij||ab⟩/ (ϵi + ϵj - ϵa - ϵb)
   # E_mp2      = 1/4 * ∑ijab ⟨ij||ab⟩ t[a,b,i,j]
   emp2 = 0.0
   @inbounds (
   for i in occupied, a in virtual,
                 j in occupied, b in virtual
      t2[a,b,i,j] += eri_smo[i,j,a,b] / (fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])
      emp2 += 0.5*0.5*t2[a,b,i,j]*eri_smo[i,j,a,b]
   end )
   t1, t2, emp2
end

###################################################
#3: Calculate the CC Intermediates
###################################################

# Effective doubles
function tau_(t1::Array{T,2},
              t2::Array{T,4},a::Int,b::Int,i::Int,j::Int) where T <: AbstractFloat
   # Stanton1991 (9)
   t2[a,b,i,j] + 0.5(t1[a,i]*t1[b,j] - t1[b,i]*t1[a,j])
end
function tau(t1::Array{T,2},
              t2::Array{T,4},a::Int,b::Int,i::Int,j::Int) where T <: AbstractFloat
   # Stanton1991 (10)
   t2[a,b,i,j] + t1[a,i]*t1[b,j] - t1[b,i]*t1[a,j]
end

function updateF_W!(Fae::Array{T,2},    # one-particle CC intermediates
                    Fmi::Array{T,2},    # one-particle CC intermediates
                    Fme::Array{T,2},    # one-particle CC intermediates
                    Wmnij::Array{T,4},  # two-particle CC intermediates
                    Wabef::Array{T,4},  # two-particle CC intermediates
                    Wmbej::Array{T,4},  # two-particle CC intermediates
                    occupied,virtual,   # occ and virt spin-MO spaces
                    t1::Array{T,2},     # CC amplitudes
                    t2::Array{T,4},     # CC amplitudes
                    fs::Diagonal{T,Array{T,1}}, # Fock matrix in spin-MO basis
                    eri_smo::Array{T,4}         # ERIs        in spin-MO basis
                   ) where T <: AbstractFloat
   nsmo = length(occupied) + length(virtual)
   # Stanton1991 (3)
   @inbounds (
   for a in virtual, e in virtual
      Fae[a,e] = (1 - (a == e))*fs[a,e]
      for m in occupied
         Fae[a,e] += -0.5*fs[m,e]*t1[a,m]
         for f in virtual
            Fae[a,e] += t1[f,m]*eri_smo[m,a,f,e]
            for n in occupied
               Fae[a,e] += -0.5*tau_(t1,t2,a,f,m,n)*eri_smo[m,n,e,f]
            end
         end
      end
   end )
   # Stanton1991 (4)
   @inbounds (
   for m in occupied, i in occupied
      Fmi[m,i] = (1 - (m == i))*fs[m,i]
      for e in virtual
         Fmi[m,i] += 0.5t1[e,i]*fs[m,e]
         for n in occupied
            Fmi[m,i] += t1[e,n]*eri_smo[m,n,i,e]
            for f in virtual
               Fmi[m,i] += 0.5tau_(t1,t2,e,f,i,n)*eri_smo[m,n,e,f]
            end
         end
      end
   end )
   # Stanton1991 (5)
   @inbounds (
   for e in virtual, m in occupied
      Fme[m,e] = fs[m,e]
      for f in virtual, n in occupied
         Fme[m,e] += t1[f,n]*eri_smo[m,n,e,f]
      end
   end )
   # Stanton1991 (6)
   @inbounds (
   for m in occupied, n in occupied, i in occupied, j in occupied
      Wmnij[m,n,i,j] = eri_smo[m,n,i,j]
      for e in virtual # P_(ij)
         Wmnij[m,n,i,j] += t1[e,j]*eri_smo[m,n,i,e] -
                           t1[e,i]*eri_smo[m,n,j,e]
         for f in virtual
            Wmnij[m,n,i,j] += 0.25tau(t1,t2,e,f,i,j)*eri_smo[m,n,e,f]
         end
      end
   end )
   # Stanton1991 (7)
   @inbounds (
   for a in virtual, b in virtual, e in virtual, f in virtual
      Wabef[a,b,e,f] = eri_smo[a,b,e,f]
      for m in occupied # P_(ab)
         Wabef[a,b,e,f] += -t1[b,m]*eri_smo[a,m,e,f] +
                            t1[a,m]*eri_smo[b,m,e,f]
         for n in occupied
            Wabef[a,b,e,f] += 0.25tau(t1,t2,a,b,m,n)*eri_smo[m,n,e,f]
         end
      end
   end )
   # Stanton1991 (8)
   @inbounds (
   for b in virtual, m in occupied, e in virtual, j in occupied
      Wmbej[m,b,e,j] = eri_smo[m,b,e,j]
      for f in virtual
         Wmbej[m,b,e,j] += t1[f,j]*eri_smo[m,b,e,f]
      end
      for n in occupied
         Wmbej[m,b,e,j] += -t1[b,n]*eri_smo[m,n,e,j]
         for f in virtual
            Wmbej[m,b,e,j] += -(0.5*t2[f,b,j,n] + t1[f,j]*t1[b,n])*eri_smo[m,n,e,f]
         end
      end
   end )
   Fae, Fmi, Fme, Wmnij, Wabef, Wmbej
end

###############################################
#4: Compute the Updated Cluster Amplitudes
###############################################

# Stanton1991 (1)
function updateT1(occupied,virtual,    # occ and virt spin-MO spaces
                  t1::Array{T,2},      # CC amplitudes
                  t2::Array{T,4},      # CC amplitudes
                  fs::Diagonal{T,Array{T,1}}, # Fock matrix in spin-MO basis
                  eri_smo::Array{T,4},        # ERIs        in spin-MO basis
                  Fae::Array{T,2},     # one-particle CC intermediates
                  Fmi::Array{T,2},     # one-particle CC intermediates
                  Fme::Array{T,2}      # one-particle CC intermediates
                 ) where T <: AbstractFloat
   nsmo = length(occupied) + length(virtual)
   T1 = zeros(nsmo,nsmo)
   @inbounds (
   for a in virtual, i in occupied
      T1[a,i] = fs[i,a] # 1st RHS term (leading term in the expansion)
      for e in virtual
         T1[a,i] += t1[e,i]*Fae[a,e] # 2nd RHS term
      end
      for m in occupied
         T1[a,i] += -t1[a,m]*Fmi[m,i] # 3rd RHS term
         for e in virtual
            T1[a,i] += t2[a,e,i,m]*Fme[m,e] # 4th RHS term
            for f in virtual
               T1[a,i] += -0.5t2[e,f,i,m]*eri_smo[m,a,e,f] # 6th RHS term
            end
            for n in occupied
               T1[a,i] += -0.5t2[a,e,m,n]*eri_smo[n,m,e,i] # 7th RHS term
            end
         end
      end
      for f in virtual,n in occupied
         T1[a,i] += -t1[f,n]*eri_smo[n,a,i,f] # 5th RHS term
      end
      # Denominator energies, Stanton1991 (12)
      T1[a,i] /= fs[i,i] - fs[a,a] # LHS
   end )
   T1
end

# Stanton1991 (2)
function updateT2(occupied,virtual,    # occ and virt spin-MO spaces
                  t1::Array{T,2},      # CC amplitudes
                  t2::Array{T,4},      # CC amplitudes
                  fs::Diagonal{T,Array{T,1}}, # Fock matrix in spin-MO basis
                  eri_smo::Array{T,4},        # ERIs        in spin-MO basis
                  Fae::Array{T,2},     # one-particle CC intermediates
                  Fmi::Array{T,2},     # one-particle CC intermediates
                  Fme::Array{T,2},     # one-particle CC intermediates
                  Wmnij::Array{T,4},   # two-particle CC intermediates
                  Wabef::Array{T,4},   # two-particle CC intermediates
                  Wmbej::Array{T,4}    # two-particle CC intermediates
                 ) where T <: AbstractFloat
   nsmo = length(occupied) + length(virtual)
   T2 = zeros(nsmo,nsmo,nsmo,nsmo)
   @inbounds (
   for a in virtual, i in occupied, b in virtual, j in occupied
      T2[a,b,i,j] = eri_smo[i,j,a,b] # 1st RHS (leading term in the expansion)
      for e in virtual # 2nd RHS term
         taeij, tbeij = t2[a,e,i,j], t2[b,e,i,j]
         T2[a,b,i,j] += taeij*Fae[b,e] -
                        tbeij*Fae[a,e]   # asym. permutes b <-> a
         for m in occupied # 3rd RHS term
            T2[a,b,i,j] += -0.5taeij*t1[b,m]*Fme[m,e] +
                            0.5tbeij*t1[a,m]*Fme[m,e]   # asym. permutes b <-> a
         end
      end
      for m in occupied # 4th RHS term
         tabim, tabjm = t2[a,b,i,m], t2[a,b,j,m]
         T2[a,b,i,j] += -tabim*Fmi[m,j] +
                         tabjm*Fmi[m,i]  # asym. permutes i <-> j
         for e in virtual # 5th RHS term
            T2[a,b,i,j] += -0.5tabim*t1[e,j]*Fme[m,e] +
                            0.5tabjm*t1[e,i]*Fme[m,e]  # asym. permutes i <-> -j
         end
      end
      for e in virtual # 10th RHS term
         T2[a,b,i,j] += t1[e,i]*eri_smo[a,b,e,j] -
                        t1[e,j]*eri_smo[a,b,e,i]  # asym. permutes i <-> j
         for f in virtual # 7th RHS term
            T2[a,b,i,j] += 0.5tau(t1,t2,e,f,i,j)*Wabef[a,b,e,f]
         end
      end
      for m in occupied # 11th RHS term
         T2[a,b,i,j] += -t1[a,m]*eri_smo[m,b,i,j] +
                         t1[b,m]*eri_smo[m,a,i,j]   # asym. permutes a <-> b
         for e in virtual # 8-9th RHS terms
            taeim, taejm = t2[a,e,i,m], t2[a,e,j,m]
            tbeim, tbejm = t2[b,e,i,m], t2[b,e,j,m]
            T2[a,b,i,j] +=  taeim*Wmbej[m,b,e,j] - t1[e,i]*t1[a,m]*eri_smo[m,b,e,j] +
                           -taejm*Wmbej[m,b,e,i] + t1[e,j]*t1[a,m]*eri_smo[m,b,e,i] + # i <-> j
                           -tbeim*Wmbej[m,a,e,j] - t1[e,i]*t1[b,m]*eri_smo[m,a,e,j] + #          a <-> b
                            tbejm*Wmbej[m,a,e,i] - t1[e,j]*t1[b,m]*eri_smo[m,a,e,i]   # i <-> j, a <-> b
         end
         for n in occupied # 6th RHS term
            T2[a,b,i,j] += 0.5tau(t1,t2,a,b,m,n)*Wmnij[m,n,i,j]
         end
      end
      # Denominator energies, Stanton1991 (13)
      T2[a,b,i,j] /=  fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b] # LHS
   end )
   T2
end

###########################################
#5: Check for Convergence and Iterate
###########################################

function extant_E_CCSD(occupied, virtual, fs, t1, t2, eri_smo)
   # Calculate the extant CC correlation energy
   # E_CC = ∑ia fia t[a,i] + 1/4 ∑ijab ⟨ij||ab⟩t[a,b,i,j] + 1/2 ∑ijab ⟨ij||ab⟩ t[a,i] t[b,j]
   E_CCSD = 0.0
   @inbounds (
   for a in virtual, i in occupied
      E_CCSD += fs[i,a] * t1[a,i]
      for b in virtual, j in occupied
         E_CCSD += 0.25eri_smo[i,j,a,b] * t2[a,b,i,j] +
                         0.5eri_smo[i,j,a,b] * t1[a,i] * t1[b,j]
      end
   end )
   E_CCSD
end

function ccsd(e, C, eri)

   nao      = size(C)[1]
   nsmo     = 2nao
   nelec    = mol.nelectron
   occupied =       1:nelec
   virtual  = nelec+1:nsmo
   eri_mo = ao2mo_smart(nao,C,eri,mol)

   eri_smo, fs = orb_to_spinorb(e,eri_mo)

   t1,t2, E_MP2 = initial_amplitudes(occupied,virtual,fs,eri_smo)

   E_CCSD  = 0.0
   ΔE_CCSD = 1.0
   iteri   = 0
   itermax = 60
   Fae = zeros(nsmo,nsmo)
   Fmi = zeros(nsmo,nsmo)
   Fme = zeros(nsmo,nsmo)
   Wmnij = zeros(nsmo,nsmo,nsmo,nsmo)
   Wabef = zeros(nsmo,nsmo,nsmo,nsmo)
   Wmbej = zeros(nsmo,nsmo,nsmo,nsmo)
   println(" Iter     Ecorr(CCSD)        ΔE(CCSD)")
   println("-----   --------------   --------------")
   while ΔE_CCSD > 1e-9 && iteri < itermax
      iteri += 1
      Fae,Fmi,Fme,Wmnij,Wabef,Wmbej = updateF_W!(Fae,Fmi,Fme,Wmnij,Wabef,Wmbej,occupied,virtual,t1,t2,fs,eri_smo)
      t1 = updateT1(occupied,virtual,t1,t2,fs,eri_smo,Fae,Fmi,Fme)
      t2 = updateT2(occupied,virtual,t1,t2,fs,eri_smo,Fae,Fmi,Fme,Wmnij,Wabef,Wmbej)
      E_CCSD, oldE_CCSD = extant_E_CCSD(occupied,virtual,fs,t1,t2,eri_smo), E_CCSD
      ΔE_CCSD = abs(E_CCSD - oldE_CCSD)
      printfmt("{1:5d}   {2:13.10f}   {3:13.10f}\n",iteri,E_CCSD,ΔE_CCSD)
   end
   @assert iteri < itermax "NOT CONVERGED!!"
   E_CCSD, E_MP2
end

