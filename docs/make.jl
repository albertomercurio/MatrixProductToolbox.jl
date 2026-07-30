using MatrixProductToolbox
using Documenter

# Doctests in docstrings assume these are loaded, matching what a user would have in scope.
const DOCTEST_SETUP = quote
    using LinearAlgebra
    using Random
    using MatrixProductToolbox
end

DocMeta.setdocmeta!(MatrixProductToolbox, :DocTestSetup, DOCTEST_SETUP; recursive = true)

const PAGES = [
    "Home" => "index.md",
    "Manual" => [
        "Matrix products" => "manual/matrix_products.md",
        "Canonical forms and truncation" => "manual/canonical_forms.md",
        "Array backends" => "manual/backends.md",
    ],
    "API" => "api.md",
]

makedocs(;
    modules = [MatrixProductToolbox],
    authors = "Alberto Mercurio <alberto.mercurio96@gmail.com> and contributors",
    sitename = "MatrixProductToolbox.jl",
    format = Documenter.HTML(;
        canonical = "https://albertomercurio.github.io/MatrixProductToolbox.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = PAGES,
)

deploydocs(;
    repo = "github.com/albertomercurio/MatrixProductToolbox.jl",
    devbranch = "main",
)
