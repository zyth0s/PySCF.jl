

#FIXME: PyObject not found when pyscf only
import PySCF: scf_rhf, ccsd #, pyscf

# Workaround: repeat the imports.
using PyCall: pyimport
pyscf = pyimport("pyscf")
mp = pyimport("pyscf.mp")   # Had to import mp alone ??!
cc = pyimport("pyscf.cc")   # Had to import mp alone ??!

@testset "CCSD: H₂O" begin

   mol = pyscf.gto.Mole()
   mol.build(verbose = 0,
             basis = "sto-3g",
             atom = """8 0  0.    0
                       1 0 -0.757 0.587
                       1 0  0.757 0.587""",
            )

   eri = mol.intor("cint2e_sph", aosym="s8")
   rhf      = scf_rhf(mol,verbose=false)
   e_rhf_me, e, C = rhf.Emol, rhf.e, rhf.C
   mf = pyscf.scf.RHF(mol) # set the model
   mf.kernel()             # run RHF (populate mf with data)

   e_mp2_pyscf = mp.MP2(mf).kernel()[1]              # pure PySCF MP₂ solution

   e_ccsd_me, e_mp2_me = ccsd(e,C,eri,mol,verbose=false)
   e_ccsd_pyscf     = cc.ccsd.CC(mf).kernel()[1]

   @test isapprox(e_mp2_me, e_mp2_pyscf, atol=1e-7)
   @test isapprox(e_ccsd_me, e_ccsd_pyscf,atol=1e-4)
#   println()
#   println("            E(HF)          Ecorr(MP2)           E(MP2)         Ecorr(CCSD)  ")
#   println("        --------------   --------------   ---------------   ---------------")
#   printfmt(" ME:    {1:13.10f}    {2:13.10f}    {3:13.10f}    {4:13.10f}\n",
#                    e_rhf_me,      e_mp2_me3,     e_rhf_me+e_mp2_me3,    e_ccsd_me)
#   printfmt(" PySCF: {1:13.10f}    {2:13.10f}    {3:13.10f}    {4:13.10f}\n",
#                     e_rhf_pyscf,  e_mp2_pyscf,e_rhf_pyscf+e_mp2_pyscf,e_ccsd_pyscf)
end
