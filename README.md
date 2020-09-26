# FiredrakeMLMC

Code base for general and parallelisable MLMC framework.

## Files
* <code>MLMCv6.py</code> - The framework. <code>MLMC_Problem</code> and <code>MLMC_Solver</code> objects should be 
            initilaised by the user. Doc-strings in this file detail the
            necessary inputs.

* <code>test_balldrop.py</code> - Example problem not using Firedrake. MLMC launched with
                    <code>general_test</code> method.

* <code>test_gaussian.py</code> - Example problem using scalar samples which very height
                    of Gaussian field centred at (0.5, 0.5). MLMC launched
                    with <code>general_test</code> method.

* <code>test_randomfield.py</code> - Example problem implementing problem shown in paper
                        by Croci et al. (doi: [10.1137/18M1175239](https://epubs.siam.org/doi/abs/10.1137/18M1175239?mobileUi=0)). Serial and
                        parallel MLMC launched with <code>general_test_serial</code>
                        and <code>general_test_para</code> methods, respectively.

* <code>Old_Files</code> - Directory containing earlier implementations of the framework.

* <code>saved_results</code> - Directory contianing .json files which store data from
                                previous runs.

## Dependencies 
Dependencies of framework found in <code>MLMCv6.py</code>:
```python
from mpi4py import MPI
import numpy as np
import time
import logging
from inspect import signature
```
