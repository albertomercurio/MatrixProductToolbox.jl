using MatrixProductToolbox
using Documenter

DocMeta.setdocmeta!(MatrixProductToolbox, :DocTestSetup, :(using MatrixProductToolbox); recursive=true)

makedocs(;
    modules=[MatrixProductToolbox],
    authors="Alberto Mercurio <alberto.mercurio96@gmail.com> and contributors",
    sitename="MatrixProductToolbox.jl",
    format=Documenter.HTML(;
        canonical="https://albertomercurio.github.io/MatrixProductToolbox.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/albertomercurio/MatrixProductToolbox.jl",
    devbranch="main",
)
