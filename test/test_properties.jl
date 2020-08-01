

#FIXME: PyObject not found when pyscf only
import PySCF: scf_rhf, dipole_moment #, pyscf

# Workaround: repeat the imports.
using PyCall: pyimport
pyscf = pyimport("pyscf")

@testset "SCF RHF dipole: H₂O" begin
   mol = pyscf.gto.Mole()
   mol.build(verbose = 0,
             basis = "sto-3g",
             atom = """8 0  0.    0
                       1 0 -0.757 0.587
                       1 0  0.757 0.587""",
            )

   mf = pyscf.scf.RHF(mol)

   rhf         = scf_rhf(mol,verbose=false)
   e_rhf_pyscf = mf.kernel()

   C = rhf.C
   nocc = mol.nelectron ÷ 2
   mocc = C[:,1:nocc]
   P  = mocc * mocc'
   dipole = dipole_moment(2P,mol)
   mulliken_pyscf, dipole_pyscf = mf.analyze(verbose=0,ncol=10, digits=9)
   @show dipole
   @show dipole_pyscf
   @test isapprox(dipole, dipole_pyscf, atol=1e-5)
end
