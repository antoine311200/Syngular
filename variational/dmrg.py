from syngular.tensor import MatrixProductOperator, MatrixProductState

from opt_einsum import contract

class DMRG:


    # Does not work with operator with input_shape != output_shape!
    # To remedy use rho = AA^t the d ensity matrix of the operator

    def solve(operator: MatrixProductOperator):

        n = operator.sites_number
        input_shape = operator.input_shape
        bond_shape = operator.bond_shape
        
        state = MatrixProductState.random(input_shape=input_shape, bond_shape=bond_shape)
        state.right_orthonormalization()

        print('[DMRG] Initialization')

        left_blocks = DMRG.__left_blocks(operator=operator, state=state)
        right_blocks = DMRG.__right_blocks(operator=operator, state=state)

        print([r.shape for r in right_blocks])
        print([r.shape for r in left_blocks])


    def __right_blocks(operator: MatrixProductOperator, state: MatrixProductState):
        right_blocks = []
        n = operator.sites_number

        print('[DMRG] Right blocks')

        for k in range(n-1,1,-1):

            state_site = state.sites[k]
            op_site = operator.sites[k]

            struct = [
                state_site, [1,3,2],
                op_site, [4,6,3,5],
                state_site, [7, 6, 8]
            ]
            if k != n-1: struct += [right_blocks[n-k-2], [2,5,8]]
            struct += [[1, 4, 7]]
            
            block = contract(*struct)
            right_blocks.append(block)
        
        return right_blocks


    def __left_blocks(operator: MatrixProductOperator, state: MatrixProductState):
        left_blocks = []
        n = operator.sites_number

        print('[DMRG] Left blocks')

        for k in range(n-2):

            state_site = state.sites[k]
            op_site = operator.sites[k]

            struct = [
                state_site, [1,3,2],
                op_site, [4,6,3,5],
                state_site, [7, 6, 8]
            ]
            if k != 0: struct += [left_blocks[k-1], [1, 4, 7]]
            struct += [[2, 5, 8]]
            
            block = contract(*struct)
            left_blocks.append(block)
        
        return left_blocks