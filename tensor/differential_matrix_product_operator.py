from turtle import right
from syngular.tensor import MatrixProductOperator, MatrixProductState

from opt_einsum import contract
import numpy as np

from typing import Union, List, Any

class DifferentialMatrixProductOperator(MatrixProductOperator):

    def gradient(self, loss):
        pass

    def crumble_site(self, index: int) -> Any:
        left_site = self.sites[index]
        right_site = self.sites[index+1]

        print(left_site.shape)
        print(right_site.shape)

        crumbled_site = contract(*[
            left_site, [1, 2, 3, 4],
            right_site, [4, 5, 6, 7],
            [1, 2, 3, 5, 6, 7]
        ])

        return crumbled_site


    """project method of DifferentialMatrixProductOperator
    Project the wings based on a specified merged site

    Mathematically, we aim at differentiating Y = WX with W a MatrixProductOperator and X an MatrixProductState
    with respect to a merge site M at index k:
        Y = WX = M(b_k, i_k, i_k+1, o_k, o_k+1, b_k+1) Z(_, _, o_1, b_k, b_k+1, o_2, _, _)
        dY/dM = Z

            |   |   |   |        |   |         |   |   |   |
        ---O---O---O---O---   ---O---O---   ---O---O---O---O---
            |   |   |   |        |   |         |   |   |   |
        ---O---O---O---O---   ---O---O---   ---O---O---O---O---

        => 
             |                        |
        ---[  ]---     |   |     ---[  ]---
        ---[  ]--------O---O--------[  ]---

        +
                       |   |
                  ---[      ]--- (6 dimensions)
                      |   |

        =>
           ||  |   |   ||
        ---O---O---O---O---     (8 dimensions)

        => (merge output and bond on wings)
           |   |   |   |
        ---O---O---O---O---     (8 dimensions)


        (Generalized Product State (not implemented yet!) of shape :
            [1, (o_1, b_k), bi_1]
            [bi_1, o_k, bi_k]
            [bi_k, o_k, bi_k+1]
            [bi_k+1, (o_2, b_k+1), 1]
        )


    @type index: int
    @param index: the index of the site n°index to merge with n°index+1

    @type state: MatrixProductState
    @param state: an input matrix product state to project with the sites

    @rtype: Mapping[str, np.ndarray]
    @returns: a dictionary containing the left/right wings, left/right centers and the merged sites
    """
    def project(self, index, state):
        
        if not isinstance(state, MatrixProductState):
            raise Exception("projected wings should come from a matrix product state input")

        if not (0 <= index < self.sites_number-1):
            raise Exception("trying to project on non-existant site (site indices should be between 0 and the number of sites - 1)")

        crumbled_site = self.crumble_site(index)

        print(crumbled_site.shape)

        """ Left wing

            (blank, blank, output, ..., output, mps_bond, mpo_bond)
               |   |   |   |
            ---O---O---O---O---
               |   |   |   |
            ---O---O---O---O---

            => 

            (blank, output*...*output_n, mps_bond, mpo_bond)
                   |
            ---[      ]===
        """

        struct = []
        n = (index+1)

        for idx in range(index):
            struct += [state.sites[idx], [ idx+1, n + idx+1, idx+2]]
            struct += [self.sites[idx], [ 2*n+idx, n+idx+1, 3*n+idx, 2*n+idx+1]]

        struct += [[1, 2*n] + list(range(3*n, 4*n-1)) + [n, 3*n-1]]

        raw_left_wing = contract(*struct)
        squeezed_left_wing = np.squeeze(raw_left_wing, axis=0)

        left_wing = np.transpose(squeezed_left_wing, axes=(list(range(1, n)) + [0, n, n+1]))
        leftover_shape = left_wing.shape[n-1:]
        left_wing = np.reshape(left_wing, newshape=(-1, *leftover_shape))
        left_wing = np.transpose(left_wing, axes=(1, 0, 2, 3))

        print("Left wing", left_wing.shape)

        
        """ Right wing

            (mps_bond, mpo_bond, output, ..., output, blank, blank)
               |   |   |   |
            ---O---O---O---O---
               |   |   |   |
            ---O---O---O---O---

            => 

            (mps_bond, mpo_bond, output*...*output_n, blank)
                   |
            ===[      ]---
        """
        
        struct = []
        n = (self.sites_number - index - 1)

        for jdx in range(index+2, self.sites_number):
            idx = jdx-index-2
            struct += [state.sites[jdx], [ idx+1, n + idx+1, idx+2]]
            struct += [self.sites[jdx], [ 2*n+idx, n+idx+1, 3*n+idx, 2*n+idx+1]]

        struct += [[1, 2*n] + list(range(3*n, 4*n-1)) + [n, 3*n-1]]

        raw_right_wing = contract(*struct)
        squeezed_right_wing = np.squeeze(raw_right_wing, axis=-1)
        
        right_wing = np.transpose(squeezed_right_wing, axes=(list(range(2, n+1)) + [0, 1, n+1]))
        leftover_shape = right_wing.shape[n-1:]
        right_wing = np.reshape(right_wing, newshape=(-1, *leftover_shape))
        right_wing = np.transpose(right_wing, axes=(1, 2, 0, 3))
        
        print("Right wing", right_wing.shape)

        left_center = state.sites[index]
        right_center = state.sites[index+1]

        print("Left Center", left_center.shape)
        print("Right Center", right_center.shape)

        return {
            'center_site': crumbled_site,
            'left_wing': left_wing,
            'left_center': left_center,
            'right_center': right_center,
            'right_wing': right_wing 
        }