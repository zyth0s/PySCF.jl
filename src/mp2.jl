
#########################################################################
# Project #4: The Second-Order Moller-Plesset
#             Perturbation Theory (MP₂) Energy
#########################################################################

using PyCall: pyimport

########################################################
#3: TRANSFORM THE TWO-ELECTRON INTEGRALS TO THE MO BASIS
########################################################

function ao2mo_noddy(nao,C,eri,mol)
   eri_mo = zeros(length(eri))
   @inbounds (
   for q in 1:nao, p in q:nao
      for r in 1:p
         lim = ifelse(p == r, q, r)
         for s in 1:lim
            pqrs = get_4idx(p,q,r,s)
            for i in 1:nao, j in 1:nao, k in 1:nao, l in 1:nao
               ijkl = get_4idx(i,j,k,l)
               eri_mo[pqrs] += C[i,p]*C[j,q]*eri[ijkl]*C[k,r]*C[l,s]
            end
         end
      end
   end )
   eri_mo
end

function ao2mo_smart(nao,C,eri,mol)
   """
   (ij|kl) => (ij|ks) => (ij|rs) => (iq|rs) => (pq|rs)
   """
   M      = nao*(nao+1)÷2
   X = zeros(nao, nao)
   tmp = zeros(M,M)
   @inbounds (
   for j in 1:nao, i in j:nao
      ij = index(i,j)
      for l in 1:nao, k in l:nao
         ijkl = get_4idx(i,j,k,l)
         X[k,l] = X[l,k] = eri[ijkl]
      end

      Y = zeros(nao, nao)
      Y = C' * X
      X = zeros(nao, nao)
      X = Y * C
      for l in 1:nao, k in l:nao
         kl = index(k,l)
         tmp[kl,ij] = X[k,l]
      end
   end )

   eri_mo = zeros(M*(M+1)÷2)

   @inbounds (
   for l in 1:nao, k in l:nao
      kl = index(k,l)
      X = zeros(nao,nao)
      Y = zeros(nao,nao)
      for j in 1:nao, i in j:nao
         ij = index(i,j)
         X[i,j] = X[j,i] = tmp[kl,ij]
      end
      Y = zeros(nao, nao)
      Y = C' *  X
      X = zeros(nao, nao)
      X = Y * C
      for j in 1:nao, i in j:nao
         klij = get_4idx(k,l,i,j)
         eri_mo[klij] = X[i,j]
      end
   end )
   eri_mo
end
#eri_mo   = pyscf.ao2mo(nao,C,eri,mol)

########################################################
#4: COMPUTE THE MP₂ ENERGY
########################################################
function short_mp2(e,C,eri,mol)
   nocc  = mol.nelectron ÷ 2
   nao   = mol.nao_nr() |> n -> convert(Int,n) # why not converted ??!
   # Convert AO -> MO basis
   eri_mo = zeros(nao,nocc,nao,nocc)
   emp2 = 0.0
   occupied =      1:nocc
   virtual  = nocc+1:nao
   pyscf = pyimport("pyscf")
   np = pyimport("numpy") # alternative: TensorOperations.jl, Einsum.jl,
   eri = pyscf.ao2mo.restore(1,eri,nao) # reshape
   eri_mo = np.einsum("pa,qi,rb,sj,pqrs->aibj",C[:,1:nocc],C,C[:,1:nocc],C,eri,optimize=true)
   @inbounds (
   for i in occupied, a in virtual,
                 j in occupied, b in virtual
       eiajb, eibja = eri_mo[i,a,j,b],  eri_mo[i,b,j,a]
       emp2 += eiajb * (2.0eiajb-eibja) / (e[i]+e[j]-e[a]-e[b])
   end )
   emp2
end

function mymp2(e,C,eri,mol; alg::Symbol=:ao2mo_smart)
   # alg ∈ [ao2mo_noddy, ao2mo_smart]
   ao2mo    = getfield(@__MODULE__, alg) # get function
   nocc     = mol.nelectron ÷ 2
   nao      = mol.nao_nr()       |> n -> convert(Int,n) # why not converted ??!
   emp2     = 0.0
   occupied =      1:nocc
   virtual  = nocc+1:nao
   eri_mo   = ao2mo(nao,C,eri,mol)
   @inbounds (
   for i in occupied, a in virtual,
                 j in occupied, b in virtual
      iajb = get_4idx(i,a,j,b)
      ibja = get_4idx(i,b,j,a)
      emp2 += eri_mo[iajb] * (2.0eri_mo[iajb]-eri_mo[ibja]) / (e[i]+e[j]-e[a]-e[b])
   end )

   # In the basis of spin-molecular orbitals:
   #emp2 += 0.25eri_mo[ijab]*eri_mo[ijab]/(e[i]+e[j]-e[a]-e[b])
   emp2
end

