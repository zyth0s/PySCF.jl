
using Formatting: printfmt
using LinearAlgebra: Diagonal, Hermitian, eigen, norm, tr, diag, dot

using PyCall: pyimport
pyscf = pyimport("pyscf")
mp = pyimport("pyscf.mp")   # Had to import mp alone ??!
cc = pyimport("pyscf.cc") # Had to import mp alone ??!

import PySCF: scf_rhf, mymp2, short_mp2, ccsd

#########################################################################
# Benchmark own HF, MP₂, and CCSD against PySCF
function benchmark_h2o_sto3g()
   mol = pyscf.gto.Mole()
   mol.build(verbose = 0,
             basis = "sto-3g",
             #atom = pyscf_atom_from_xyz("../data/h2.xyz"),
             #atom = pyscf_atom_from_xyz("../data/acetaldehyde.xyz"),
             atom = """8 0  0.    0
                       1 0 -0.757 0.587
                       1 0  0.757 0.587""",
            )

   mf = pyscf.scf.RHF(mol)

   rhf = scf_rhf(mol,verbose=false)
   e_rhf_me, e, C = rhf.Emol, rhf.e, rhf.C
   e_rhf_pyscf    = mf.kernel()

   #mf.analyze(verbose=3,ncol=10, digits=9)

   println( "           E(HF)     ")
   println( "        -------------")
   printfmt(" ME:    {1:13.8f}  \n", e_rhf_me)
   printfmt(" PySCF: {1:13.8f}  \n", e_rhf_pyscf)


   eri   = mol.intor("cint2e_sph", aosym="s8")
   println()
   e_mp2_me1   = @time mymp2(e,C,eri,mol,alg=:ao2mo_noddy) # Naive loop  ao2mo with my MP₂
   e_mp2_me2   = @time mymp2(e,C,eri,mol,alg=:ao2mo_smart) # Smart loop  ao2mo with my MP₂
   e_mp2_me3   = @time short_mp2(e,C,eri,mol)              # @tensor     ao2mo with my MP₂
   e_mp2_pyscf = @time mp.MP2(mf).kernel()[1]              # pure PySCF MP₂ solution
   println()

   println( "           Ecorr(MP₂)        E(MP₂)   ")
   println( "          -----------   -------------")
   printfmt(" ME1:   {1:13.8f}   {2:13.8f}  \n", e_mp2_me1,  e_rhf_me   +e_mp2_me1)
   printfmt(" ME2:   {1:13.8f}   {2:13.8f}  \n", e_mp2_me2,  e_rhf_me   +e_mp2_me1)
   printfmt(" ME3:   {1:13.8f}   {2:13.8f}  \n", e_mp2_me3,  e_rhf_me   +e_mp2_me1)
   printfmt(" PySCF: {1:13.8f}   {2:13.8f}  \n", e_mp2_pyscf,e_rhf_pyscf+e_mp2_pyscf)
   println()

   @time e_ccsd_me, emp2 = ccsd(e,C,eri,mol,verbose=false)
   @time e_ccsd_pyscf    = cc.ccsd.CC(mf).kernel()[1]
   emp2 ≈ e_mp2_me3 || @warn "E_MP2 from CC is not accurate"
   println()

   println("            E(HF)          Ecorr(MP2)           E(MP2)         Ecorr(CCSD)  ")
   println("        --------------   --------------   ---------------   ---------------")
   printfmt(" ME:    {1:13.10f}    {2:13.10f}    {3:13.10f}    {4:13.10f}\n",
                    e_rhf_me,      e_mp2_me3,     e_rhf_me+e_mp2_me3,    e_ccsd_me)
   printfmt(" PySCF: {1:13.10f}    {2:13.10f}    {3:13.10f}    {4:13.10f}\n",
                     e_rhf_pyscf,  e_mp2_pyscf,e_rhf_pyscf+e_mp2_pyscf,e_ccsd_pyscf)
end

benchmark_h2o_sto3g()
