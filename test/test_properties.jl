

#FIXME: PyObject not found when pyscf only
import PySCF: scf_rhf #, pyscf
import PySCF: dipole_moment
import PySCF: population_analysis
import PySCF: number_pairs_j, number_pairs_xc, number_pairs

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
   dipole_pyscf = mf.analyze(verbose=0,ncol=10, digits=9)[2]
   @test isapprox(dipole, dipole_pyscf, atol=1e-5) # [Debye]
end

@testset "SCF RHF Mulliken analysis: H₂O" begin
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

   s     = mol.intor("cint1e_ovlp_sph")

   C = rhf.C
   nocc = mol.nelectron ÷ 2
   mocc = C[:,1:nocc]
   P  = mocc * mocc'
   # In RHF, D = 2P, an occupied spatial orbital means 2 elec, α and β
   # PS is a "correct" representation of the 1RDM in AO basis
   Mulliken = 1; #Lowdin = 0.5
   QMulliken = population_analysis(2P,s,Mulliken,mol,verbose=false)

   QMulliken_pyscf = mf.analyze(verbose=0,ncol=10, with_meta_lowdin=false, digits=9)[1][2]
   @test isapprox(QMulliken, QMulliken_pyscf, atol=1e-7) # a.u.
end

@testset "SCF RHF Number of electron pairs: H₂O" begin
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

   s     = mol.intor("cint1e_ovlp_sph")

   C = rhf.C
   nocc = mol.nelectron ÷ 2
   mocc = C[:,1:nocc]
   P  = mocc * mocc'
   # In RHF, D = 2P, an occupied spatial orbital means 2 elec, α and β
   # PS is a "correct" representation of the 1RDM in AO basis

   s = mol.intor("int1e_ovlp")
   pairs_j  = number_pairs_j(2P,s)
   pairs_xc = number_pairs_xc(2P,s)
   pairs    = number_pairs(2P,s)
   nelec = mol.nelectron
   @test pairs_j ≈ nelec^2
   @test pairs_xc ≈ nelec
   @test pairs ≈ nelec*(nelec-1)
end
