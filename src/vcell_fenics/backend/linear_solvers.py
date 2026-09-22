"""Choosing a PETSc preconditioner that works at any MPI rank count.

The method-of-lines integrators default to GMRES + ILU (VCell's CVODE uses SPGMR + ILU). PETSc's ILU is
**sequential**: on a distributed (``mpiaij``) matrix ``PCSetUp`` fails with "Could not locate a solver type
for factorization type ILU and matrix type mpiaij" — found by the first MPI smoke run of the container
(integration step 10). PETSc's standard parallel stand-in is **block Jacobi**, whose default sub-solver
is an ILU(0) of each rank's diagonal block, so :func:`set_preconditioner` makes that substitution when the
matrix is distributed and leaves a serial run exactly as it was (bjacobi on one block *is* ILU, but the
serial path is kept byte-for-byte unchanged).

A direct ``lu`` needs no substitution: in parallel PETSc picks a distributed factorization package
(MUMPS in the conda-forge build) on its own.
"""

from __future__ import annotations

from petsc4py import PETSc


def set_preconditioner(ksp: PETSc.KSP, pc_type: str) -> None:
    """``ksp.getPC().setType(pc_type)``, with ILU replaced by block-Jacobi/ILU(0) on more than one rank."""

    parallel = ksp.getComm().getSize() > 1
    ksp.getPC().setType("bjacobi" if parallel and pc_type == "ilu" else pc_type)
