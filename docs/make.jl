using TTSolver
using Documenter

DocMeta.setdocmeta!(TTSolver, :DocTestSetup, :(using TTSolver); recursive=true)

makedocs(;
    modules=[TTSolver],
    authors="htkhsh",
    sitename="TTSolver.jl",
    format=Documenter.HTML(;
        canonical="https://takahashi.github.io/TTSolver.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/takahashi/TTSolver.jl",
    devbranch="main",
)
