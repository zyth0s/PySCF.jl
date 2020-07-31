using Test

import PySCF: get_4idx

@testset "PySCF.jl" begin
    # Write your tests here.
    @test get_4idx(1,2,3,4) == 38
end
