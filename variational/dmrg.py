from syngular.tensor import MatrixProductOperator, MatrixProductState
from syngular.variational.optimizer import Optimizer, Lanczos

import numpy as np
from opt_einsum import contract

from typing import Type


class DMRG:


    # Does not work with operator with input_shape != output_shape!
    # To remedy use rho = AA^t the d ensity matrix of the operator

    def solve(operator: MatrixProductOperator, optimizer: Type[Optimizer]=Lanczos):

        n = operator.sites_number
        input_shape = operator.input_shape
        bond_shape = operator.bond_shape
        
        state = MatrixProductState.random(input_shape=input_shape, bond_shape=bond_shape)
        state.right_orthonormalization()

        print('[DMRG] Initialization')

        left_blocks = DMRG.__left_blocks(operator=operator, state=state)
        right_blocks = DMRG.__right_blocks(operator=operator, state=state)

        print([r.shape for r in right_blocks if r is not None])
        print([r.shape for r in left_blocks if r is not None])


        print('[DMRG] Right sweep')
        DMRG.__right_sweep(
            operator=operator, state=state, 
            left_blocks=left_blocks, right_blocks=right_blocks, 
            optimizer=optimizer
        )

        print('[DMRG] Left sweep')
        DMRG.__left_sweep(
            operator=operator, state=state, 
            left_blocks=left_blocks, right_blocks=right_blocks, 
            optimizer=optimizer
        )

        return state

    def __right_blocks(operator: MatrixProductOperator, state: MatrixProductState):
        n = operator.sites_number
        right_blocks = [None for _ in range(n)]

        print('[DMRG] Right blocks')

        for k in range(n-1,1,-1):

            state_site = state.sites[k]
            op_site = operator.sites[k]

            struct = [
                state_site, [1,3,2],
                op_site, [4,6,3,5],
                state_site, [7, 6, 8]
            ]
            if k != n-1: struct += [right_blocks[k+1], [2,5,8]]
            struct += [[1, 4, 7]]
            
            block = contract(*struct)
            right_blocks[k] = block
        
        return right_blocks


    def __left_blocks(operator: MatrixProductOperator, state: MatrixProductState):
        n = operator.sites_number
        left_blocks = [None for _ in range(n)]

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
            left_blocks[k] = block
        
        return left_blocks


    '''

    
               |   |
            ---O---O---   ---[ ]
               |   |         [ ]
                          ---[ ]
               |  |          [ ]
            --[   ]--     ---[ ] (R3)
              (B12)

    '''
    def __right_sweep(operator: MatrixProductOperator, state: MatrixProductState, left_blocks: list, right_blocks: list, optimizer: Type[Optimizer]):
        n = operator.sites_number
        
        for k in range(n-1):
            print(f"Step {k+1}/{n-1}")
            merge = contract(state.sites[k], [1,2,3], state.sites[k+1], [3,4,5])

            # print("Shape start step", [site.shape for site in state.sites])

            print(f"Merge site shape [idx {k}]", merge.shape)

            if k == 0:
                site = contract(
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    right_blocks[k+2], [10,8,4],
                    [1, 7, 9, 10]
                )
            elif k == n-2:
                site = contract(
                    left_blocks[k-1], [10,5,1],
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    [10, 7, 9, 4]
                )
            else:
                site = contract(
                    left_blocks[k-1], [11,5,1],
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    right_blocks[k+2], [10,8,4],
                    [11, 7, 9, 10]
                )
            
            print(f"Contracted state shape [idx {k}]", merge.shape)

            nsite = optimizer.fit(site)
            state.sites[k], state.sites[k+1] = DMRG.__restore(state=state, site=nsite, index=k)

            # print("Shape end step", [site.shape for site in state.sites])

    
    def __left_sweep(operator: MatrixProductOperator, state: MatrixProductState, left_blocks: list, right_blocks: list, optimizer: Type[Optimizer]):
        n = operator.sites_number
        
        for k in range(n-2, -1, -1):
            print(f"Step {n-1-k}/{n-1}")

            merge = contract(state.sites[k], [1,2,3], state.sites[k+1], [3,4,5])

            print(f"Merge site shape [idx {k}]", merge.shape)

            if k == 0:
                site = contract(
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    right_blocks[k+2], [10,8,4],
                    [1, 7, 9, 10]
                )
            elif k == n-2:
                site = contract(
                    left_blocks[k-1], [10,5,1],
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    [10, 7, 9, 4]
                )
            else:
                site = contract(
                    left_blocks[k-1], [11,5,1],
                    merge, [1,2,3,4], 
                    operator.sites[k], [5,2,7,6],
                    operator.sites[k+1], [6,3,9,8],
                    right_blocks[k+2], [10,8,4],
                    [11, 7, 9, 10]
                )
            
            print(f"Contracted state shape [idx {k}]", merge.shape)

            nsite = optimizer.fit(site)
            state.sites[k], state.sites[k+1] = DMRG.__restore(state=state, site=nsite, index=k)


    def __restore(state: MatrixProductState, site: np.ndarray, index: int) -> tuple[np.ndarray]:
        
        if index >= state.sites_number:
            raise Exception("site index for restoration is out of range (between 0 & n-1)")
        
        L = np.reshape(site, newshape=(site.shape[0]*site.shape[1], -1))
        Q, R = np.linalg.qr(L, mode="complete")

        # print("L", L.shape, "Q", Q.shape, "R", R.shape)

        rank = state.shape[index][2]
        Q = Q[:,:rank]
        R = R[:rank, :]

        site1 = np.reshape(Q, newshape=state.shape[index])
        site2 = np.reshape(R, newshape=state.shape[index+1])

        return site1, site2
