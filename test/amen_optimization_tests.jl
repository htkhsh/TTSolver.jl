@testset "pre-migration AMEn reference" begin
    Random.seed!(20261010)
    @test relative_l2(run_amen(:direct), AMEN_DIRECT_REFERENCE) <= 10eps(Float64)

    Random.seed!(20261010)
    @test relative_l2(run_amen(:iterative), AMEN_ITERATIVE_REFERENCE) <= 10eps(Float64)
end
