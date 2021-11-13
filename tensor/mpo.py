from functools import reduce
from typing import Union, Any


import numpy as np
from numpy.core.einsumfunc import einsum


def unfold_shape(shape):
    return reduce(lambda x, y: x+y, shape)

def unfold_dim(shape):
    return reduce(lambda x, y: x*y, shape)

class MatrixProductOperator:

    def __init__(self, tensor, input_shape, output_shape, bond_shape) -> None:
        self.tensor = tensor

        self.input_shape = input_shape
        self.unfold_input_shape = unfold_shape(self.input_shape)
        self.output_shape = output_shape
        self.unfold_output_shape = unfold_shape(self.output_shape)
        self.bond_shape = bond_shape
        self.unfold_bond_shape = unfold_shape(self.bond_shape)

        self.index_length = len(input_shape)+len(output_shape)+len(bond_shape)

        self.cores_number = len(input_shape)

        self.cores = []
        self.__bond_core_index = None

    def retrieve(self, input_indices, output_indices):
        einsum_structure = []
        for idx in range(self.cores_number):
            print(self.cores[idx][input_indices[idx], output_indices[idx]])
            einsum_structure.append(self.cores[idx][input_indices[idx], output_indices[idx]])
            if idx == 0:
                einsum_structure.append([Ellipsis, idx])
            elif idx == self.cores_number-1:
                einsum_structure.append([Ellipsis, idx-1])
            else:
                einsum_structure.append([Ellipsis, idx-1, idx])

        print(einsum_structure)
        return np.einsum(*einsum_structure)

    def decompose(self):
        leftover = self.tensor
        
        for idx in range(self.cores_number):
            if idx != self.cores_number-1:
                core, leftover = self.__create_svd_core(leftover, idx, self.bond_shape[idx])
            else:
                core, leftover = self.__create_svd_core(leftover, idx, None)
            self.cores.append(core)

    def __create_svd_core(self, leftover, index, chi, reshape=True):

        if index != 0 and index != self.cores_number-1:
            '''
            Core shape decomposition into SVD
                      |   |   |           
                 ---[           ]   ---> (reshape)   ---O---   ---> ---O---  o---0---
                      |   |   |           
 
                     |                             |           |   |      
            --->  ---O---   o---O---    --->    ---O---  o---[       ]  
                     |                             |           |   |     

                     |          |   |  
            --->  ---O---  ---[       ]  
                     |          |   |  

            (inp1, inp2, inp3, out1, out2, out3, left) -> ((inp1, out1, left), (inp2, inp3, out2, out3))
            (   0,    1,    2,    3,    4,    5,    6) -> ((   0,    3,    6), (   1,    2,    4,    5))
            '''


            if reshape:
                idx_length = len(leftover.shape)
                inp1, out1, left = leftover.shape[0], leftover.shape[self.cores_number-index], leftover.shape[-1]
            else:
                idx_length = len(self.input_shape[index:])+len(self.output_shape[index:])+1
                inp1, out1, left = self.input_shape[index], self.output_shape[index], self.bond_shape[index]

            left_dim = inp1 * out1 * left
            right_dim = unfold_dim(leftover.shape)//left_dim
            
            newdim = [left_dim, right_dim]

            print("Decomposition of the core n°", index+1)
            print(" |__ shape ", leftover.shape)
            print(" |__ reshaped to ", newdim)

            if reshape:
                except_axes = (0, self.cores_number-index, 2*(self.cores_number-index))
                axes = tuple(i for i in range(idx_length) if i not in except_axes)

                matrix_core_transposed = np.transpose(leftover, except_axes + axes)
                matrix_core = np.reshape(matrix_core_transposed, newdim, order='F')
                
                print(" |__ transposed axes ", except_axes + axes)
            else:
                # Need to swap first axe to be the n-1 th (left, right)
                matrix_core = np.transpose(leftover, axes=(1,0))

                print(" |__ transposed axes (1, 0)")


            u, s, v = np.linalg.svd(matrix_core, full_matrices=False)

            print(f" |__ SVD shape (U {u.shape}, S {s.shape}, V {v.shape}")
            
            u_trimmed = u[:,:chi]
            s_trimmed = s[:chi]
            v_trimmed = v[:chi,:]
            print(s, s_trimmed)

            print(f" |__ SVD trimmed with chi {chi} (U {u_trimmed.shape}, S {s_trimmed.shape}, V {v_trimmed.shape}")

            core = np.reshape(u_trimmed, newshape=(inp1, out1, left, chi), order='F')

            new_leftover_transposed = np.transpose(s_trimmed * v_trimmed.T, (1,0))
            if reshape:
                newshape_leftover = tuple(map(lambda axe: leftover.shape[axe], axes))+(chi,)
                # print(newshape_leftover, axes, axes[:-1])
            else:
                newshape_leftover = (right_dim, chi)
            new_leftover = np.reshape(new_leftover_transposed, newshape=newshape_leftover, order='F')
            
            print(" |__ core ", core.shape)
            print(" |__ leftover ", newshape_leftover)
            print("\n")

        elif index == 0:
            idx_length = len(leftover.shape)
            inp1, out1 = leftover.shape[0], leftover.shape[self.cores_number]
            
            left_dim = inp1 * out1
            right_dim = unfold_dim(leftover.shape)//left_dim
            
            newdim = [left_dim, right_dim]

            print("Decomposition of the core n°1 (START)")
            print(" |__ shape ", leftover.shape)
            print(" |__ reshaped to ", newdim)

            except_axes = (0, self.cores_number)
            axes = tuple(i for i in range(idx_length) if i not in except_axes)

            matrix_core_transposed = np.transpose(leftover, except_axes + axes)
            matrix_core = np.reshape(matrix_core_transposed, newdim, order='F')

            print(" |__ transposed axes ", except_axes + axes)

            u, s, v = np.linalg.svd(matrix_core, full_matrices=False)

            print(f" |__ SVD shape (U {u.shape}, S {s.shape}, V {v.shape}")
            
            u_trimmed = u[:,:chi]
            s_trimmed = s[:chi]
            v_trimmed = v[:chi,:]

            print(s, s_trimmed)
            print(f" |__ SVD trimmed with chi {chi} (U {u_trimmed.shape}, S {s_trimmed.shape}, V {v_trimmed.shape}")

            core = np.reshape(u_trimmed, newshape=(inp1, out1, chi), order='F')

            new_leftover_transposed = np.transpose(s_trimmed * v_trimmed.T, (1,0))
            if reshape:
                newshape_leftover = tuple(map(lambda axe: leftover.shape[axe], axes))+(chi,)
            else:
                newshape_leftover = (right_dim, chi)
            new_leftover = np.reshape(new_leftover_transposed, newshape=newshape_leftover, order='F')
            
            print(" |__ core ", core.shape)
            print(" |__ leftover ", newshape_leftover)
            print("\n")
        else:
            core = leftover
            new_leftover = []

            print(f"Decomposition of the core n°{index+1} (END)")
            print(" |__ nothing to compute")
            print(" |__ core shape ", core.shape)
            print("\n")

        return core, new_leftover
 
if __name__ == "__main__":

    # tensor = np.arange(((2**6)*(3**6))).reshape(((2,2,2,2,2,2,3,3,3,3,3,3)))

    # tt = MatrixProductOperator(tensor, (2,2,2,2,2,2), (3,3,3,3,3,3), (4,4,4,4,4))
    # tt.decompose()

    # print(tt.retrieve((0,0,0,0,0,0),(0,0,0,0,0,0)))

    tensor = np.arange(4*4).reshape((2,2,2,2))
    tt = MatrixProductOperator(tensor, (2,2), (2,2), (2,))
    tt.decompose()
    print(tt.cores)

    print(tensor)
    print(tt.retrieve((0,0),(0,0)))