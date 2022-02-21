from numpy import outer
from syngular.variational import DMRG
from syngular.tensor import MatrixProductOperator

if __name__ == "__main__":

    input_shape = (3,3,3,3,3,3)
    output_shape = (3,3,3,3,3,3)
    bond_shape = (2,2,2,2,2)

    mpo = MatrixProductOperator.random(input_shape=input_shape, output_shape=output_shape, bond_shape=bond_shape)

    DMRG.solve(mpo)