# dihedral_model

This package is a collection of code to model semiflexible polymers (e.g. polythiophene, polypyrrole) using a stochastic dihedral potential method. The dihedral potential model generates an ensemble of chain conformations, which can be described by average structural properties. The main algorithm to generate a chain conformation is implemented in core.polymer_chain as a Polymer or RandomChargePolymer object. Instead of storing all the chain conformations, running statistics (utils.stats) are employed to calculate different moments of a distribution on the fly. More examples and documentation will be added with a forthcoming publication (~ May 2019). E-mail b.wood@berkeley with questions.

## Structural Properties Currently Supported

-	End-to-end distance
-	Persistence length
-	Planarity (orientational order parameter)

