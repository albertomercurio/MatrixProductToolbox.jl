JULIA := julia
GROUP ?= Core

.PHONY: help setup test test-all format docs

help:
	@echo "Targets:"
	@echo "  setup     install Runic into the @runic shared environment"
	@echo "  test      run one test group (GROUP=$(GROUP)); see below"
	@echo "  test-all  run Core and Code-Quality in separate processes"
	@echo "  format    format the repository with Runic"
	@echo "  docs      build the documentation, including doctests"
	@echo ""
	@echo "Test groups: Core, Code-Quality, CUDA, Reactant"
	@echo "  Core          contractions, gauge, observables, CPU backend probes"
	@echo "  Code-Quality  Aqua + JET; memory hungry, so keep it in its own process"
	@echo "  CUDA          test/cuda/,     own sub-environment, needs a GPU"
	@echo "  Reactant      test/reactant/, own sub-environment, runs on CPU"

# Note the `--project=@runic` form: only the command line resolves `@name` to a *shared*
# environment. `Pkg.activate("@runic")` would create a local directory called `@runic` instead.
setup:
	$(JULIA) --startup-file=no --project=@runic -e 'using Pkg; Pkg.add("Runic")'

test:
	GROUP=$(GROUP) $(JULIA) --startup-file=no --project=. -e 'using Pkg; Pkg.test()'

# Deliberately separate processes: running JET alongside everything else is what makes the
# suite heavy.
test-all:
	$(MAKE) test GROUP=Core
	$(MAKE) test GROUP=Code-Quality

format:
	$(JULIA) --project=@runic --startup-file=no -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace .

docs:
	$(JULIA) --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()'
	$(JULIA) --startup-file=no --project=docs docs/make.jl
