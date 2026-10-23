using Test
using TTSolver

@testset "Chebyshev interpolation reproduces polynomials" begin
    for node_count in (3, 5)
        coarse, _ = chebdiff(node_count)
        fine, _ = chebdiff(2 * node_count)
        interpolation = cheb2_interpolant(coarse, fine)
        for degree in 0:(node_count - 1)
            @test interpolation * coarse.^degree ≈ fine.^degree atol=2e-13
        end
    end
end
