@testset "pre-migration TAMEn reference" begin
    Random.seed!(20261010)
    @test maximum(relative_l2.(run_tamen("cheb"), TAMEN_CHEB_REFERENCE)) <=
          10eps(Float64)

    Random.seed!(20261010)
    @test maximum(relative_l2.(run_tamen("CN"), TAMEN_CN_REFERENCE)) <=
          10eps(Float64)
end
