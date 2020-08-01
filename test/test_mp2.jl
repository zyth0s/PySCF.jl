
#FIXME: PyObject not found when pyscf only
import PySCF: scf_rhf, mymp2, short_mp2 #, pyscf

# Workaround: repeat the imports.
using PyCall: pyimport
pyscf = pyimport("pyscf")
mp = pyimport("pyscf.mp")   # Had to import mp alone ??!

@testset "MP₂: H₂O" begin

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

   e_mp2_me1   = mymp2(e,C,eri,mol,alg=:ao2mo_noddy) # Naive loops ao2mo with my MP₂
   e_mp2_me2   = mymp2(e,C,eri,mol,alg=:ao2mo_smart) # Smart loops ao2mo with my MP₂
   e_mp2_me3   = short_mp2(e,C,eri,mol)              # einsum      ao2mo with my MP₂
   e_mp2_pyscf = mp.MP2(mf).kernel()[1]              # pure PySCF MP₂ solution

   @test isapprox(e_mp2_me1, e_mp2_pyscf, atol=1e-7)
   @test isapprox(e_mp2_me2, e_mp2_pyscf, atol=1e-7)
   @test isapprox(e_mp2_me3, e_mp2_pyscf, atol=1e-7)
end
