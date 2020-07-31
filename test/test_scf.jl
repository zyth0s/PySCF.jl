
#FIXME: PyObject not found when pyscf only
import PySCF: scf_rhf #, pyscf

# Workaround: repeat the imports.
using PyCall: pyimport
pyscf = pyimport("pyscf")

@testset "SCF RHF: H₂O" begin
   mol = pyscf.gto.Mole()
   mol.build(verbose = 0,
             basis = "sto-3g",
             atom = """8 0  0.    0
                       1 0 -0.757 0.587
                       1 0  0.757 0.587""",
            )

   mf = pyscf.scf.RHF(mol)

   e_rhf_me, e, C = scf_rhf(mol,verbose=false)
   e_rhf_pyscf    = mf.kernel()

   @test isapprox(e_rhf_pyscf, e_rhf_me, atol=1e-10)
end
