## Quantum Belief Propagation

Software tool for running belief propagation on quantum models.
For 1D model use `Hamiltonian` and `BeliefPropagator`.
For 2D lattice model use `LatticeHamiltonian` and `LatticeBeliefPropagator`.

The `examples/1d/` and `examples/2d/` directories contain some code used to
produce the relevant graphs in the [thesis paper](https://gitlab.doc.ic.ac.uk/jz4120/personal-project).
If you wish to reproduce the results, for example, run
`python3 -m examples.1d.correctness`.
