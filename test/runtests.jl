using Test

import PySCF: get_4idx

@testset "Indexing functions" begin
   @test get_4idx(1,2,3,4) == 38
end


include("test_scf.jl")
include("test_mp2.jl")
include("test_ccsd.jl")
include("test_properties.jl")

