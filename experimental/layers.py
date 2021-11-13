from __future__ import print_function, annotations
from functools import reduce
from re import match

import numpy as np
from numpy.core.einsumfunc import einsum
from numpy.lib.function_base import _CORE_DIMENSION_LIST
from numpy.matrixlib.defmatrix import matrix


def unfold_shape(shape):
    return reduce(lambda x, y: x+y, shape)

class TensorTrainLayer():

    def __init__(self) -> None:
        pass

    def build(self):
        pass

    def call(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)

    def train(self):
        pass

class ReLU(TensorTrainLayer):
    
    def forward(self, input):
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(TensorTrainLayer):

    def __init__(self, input_shape, output_shape, bond_dim=2, core_number=None, learning_rate=0.01) -> None:
        
        if len(input_shape) != len(output_shape):
            raise Exception("input shape and output shape should have the same length")

        if core_number != None and core_number != len(input_shape):
            raise Exception("number of cores does not match the size of input_shape")
        
        self.input_shape = input_shape
        self.unfold_input_shape = unfold_shape(self.input_shape)
        self.output_shape = output_shape
        self.unfold_output_shape = unfold_shape(self.output_shape)

        self.cores_number = core_number if core_number != None else len(input_shape)
        self.bond_dim = bond_dim

        self.learning_rate = learning_rate

        self.cores = []
        self.bias = []

        self.__bond_core = None
        self.__bond_core_index = None
        self.__bond_core_update_step = False

    def __get_core_shape(self, index):
        if index == 0 or index == self.cores_number-1:
            return (self.input_shape[index], self.output_shape[index], self.bond_dim,)
        else:
            return (self.input_shape[index], self.output_shape[index], self.bond_dim, self.bond_dim,)

    def __add_core(self, name, type):
        index = len(self.cores)

        shape = self.__get_core_shape(index)
        size = unfold_shape(shape)

        print(shape)

        if type == 'middle' and 0 < index < self.cores_number-1:
            return np.random.normal(
                loc=0.0,
                scale = np.sqrt(2/size), 
                size = shape
            )
        elif type == 'extreme' and (index == 0 or index == self.cores_number-1):
            return np.random.normal(
                loc=0.0,
                scale = np.sqrt(2/size), 
                size = shape
            )
        else:
            raise Exception('the type of core to add does not match the current cores structure')

    def __contract_bond_core(self, index):
        if not self.__bond_core_update_step:
            raise Exception("an updating step of the cores has not been started yet")
        
        if index >= self.cores_number or index < 0:
            raise Exception(f"the index provided for contraction is out of bond ({index})")

        __copy_cores = self.cores.copy()
        current_core = __copy_cores[index]
        next_core = __copy_cores.pop(index+1)

        print(index, self.cores_number-2)
        print(current_core.shape)
        print(next_core.shape)
        if index == 0:
            __copy_cores[index] = np.einsum(current_core, [10, 11, 0], next_core, [20, 21, 0, 1], [10,20,11,21,1])
        elif index != self.cores_number-2:
            __copy_cores[index] = np.einsum(current_core, [10,11,0,1], next_core, [20,21,1,2], [10,20,11,21,0,2])
        else:
            __copy_cores[index] = np.einsum(current_core, [10, 11, 0, 1], next_core, [20, 21, 1], [10, 20, 11, 21, 0])

        print(__copy_cores[index].shape)
        self.__bond_core_index = index
        self.__bond_core = __copy_cores[self.__bond_core_index]

    def __project_wings_MPO(self, input, contracted_index):
        if not self.__bond_core_update_step:
            raise Exception("an updating step of the cores has not been started yet")

        input = np.array(input)
        unfold_input = unfold_shape(input.shape)

        if self.unfold_input_shape != unfold_input:
            exception = f"input of shape {input.shape} cannot be reshaped into {self.input_shape} [{unfold_input} != {self.unfold_input_shape}]"
            raise Exception(exception)
        
        '''
                |   |   |   |                    |           |   |           |    
             ---O---O---O---O---     --->     ---O---   ---[       ]---   ---O---
                |   |   |   |                    |           |   |           |        
                O   O   O   O                    O           O   O           O


               |             |
            ---O--- ==>   ---O---
               |
               O           


                    |           |   |           |    
            ==>  ---O---   ---[       ]---   ---O---
        '''
        input_index = list(np.arange(self.cores_number))

        einsum_structure = []
        einsum_structure.append(input)
        einsum_structure.append(input_index)

        for idx in range(self.cores_number):
            if idx != self.__bond_core_index and idx != self.__bond_core_index+1:
                einsum_structure.append(self.cores[idx])

                ipt_index = idx
                if idx == 0:
                    opt_index = self.cores_number+idx
                    right_index = 2*self.cores_number+idx
                    einsum_structure.append([ipt_index, opt_index, right_index])
                elif idx == self.cores_number-1:
                    opt_index = self.cores_number+idx
                    left_index = 2*self.cores_number+idx-1
                    einsum_structure.append([ipt_index, opt_index, left_index])
                else:
                    opt_index = self.cores_number+idx
                    left_index = 2*self.cores_number+idx-1
                    right_index = 2*self.cores_number+idx
                    einsum_structure.append([ipt_index, opt_index, left_index, right_index])
        
        output_start = list(np.arange(self.cores_number,self.cores_number+self.__bond_core_index))
        output_end = list(np.arange(self.cores_number+self.__bond_core_index+2, 2*self.cores_number))
        bond_start = list(np.arange(2*self.cores_number, 2*self.cores_number+self.__bond_core_index))
        bond_end = list(np.arange(2*self.cores_number+self.__bond_core_index+1, 3*self.cores_number-1))
        
        output_index = output_start + output_end + bond_start + bond_end

        print('Output shape', output_index)
        einsum_structure.append(output_index)

        # print(einsum_structure)

        projected_tensor = np.einsum(*einsum_structure)
        return projected_tensor

    def __project_wings_MPS(self, input, contracted_index):
        if not self.__bond_core_update_step:
            raise Exception("an updating step of the cores has not been started yet")
        
        input = np.array(input)
        unfold_input = unfold_shape(input.shape)

        if self.unfold_input_shape != unfold_input:
            exception = f"input of shape {input.shape} cannot be reshaped into {self.input_shape} [{unfold_input} != {self.unfold_input_shape}]"
            raise Exception(exception)

        input_tensor = np.reshape(input, newshape=self.input_shape)
        input_index = np.arange(self.cores_number)

        einsum_structure = []
        einsum_structure.append(input_tensor)
        einsum_structure.append(input_index)

        for idx in range(self.cores_number):
            ipt_index = idx
            opt_index = self.cores_number+idx

            if idx == contracted_index or idx == contracted_index+1:
                einsum_structure.append(self.cores[idx])
                if idx == 0:
                    bnd_index = 2*self.cores_number
                    einsum_structure.append([ipt_index, opt_index, bnd_index])
                elif idx == self.cores_number-1:
                    bnd_index = 3*self.cores_number-2
                    einsum_structure.append([ipt_index, opt_index, bnd_index])
                else:
                    bnd_index_1 = 2*self.cores_number+idx-1
                    bnd_index_2 = 2*self.cores_number+idx
                    einsum_structure.append([ipt_index, opt_index, bnd_index_1, bnd_index_2])

        projected_tensor = np.einsum(*einsum_structure)

        return projected_tensor

    def __reset_bond_core(self):
        self.__bond_core = None
        self.__bond_core_index = None
        self.__bond_core_update_step = False

    # Possible error in the tuples of axes for reshaping the tensor into a matrix for SVD
    def __SVD_bond_core(self, chi=None, verbose=False):
        if self.__bond_core_index != 0 and self.__bond_core_index != self.cores_number-2:
            '''
            Core shape decomposition with SVD:

                       |   |                      |   |       
                    ---O---O---     --->     ---[       ]--- 
                       |   |                      |   |          
 
                          |       |                  |         |                  |   |
                 --->  ---O---o---O---    --->    ---O---  o---O---    --->    ---O---O---
                          |       |                  |         |                  |   |

            # (in1, in2, out1, out2, left, right) -> ((in1, out1, left), (in1, out2, right))
            # (0,     1,    2,    3,    4,     5) -> ((  0,    2,    4), (  1,    3,     5))
            '''

            inp1, inp2, out1, out2, left, right = self.__bond_core.shape
            newdim = [inp1 * out1 * left, inp2 * out2 * right]

            matrix_core_transposed = np.transpose(self.__bond_core, (0,2,4,1,3,5))
            matrix_core = np.reshape(matrix_core_transposed, newdim, order='F')
            u, s, v = np.linalg.svd(matrix_core, full_matrices=False)

            u_trimmed = u[:,:chi]
            s_trimmed = s[:chi]
            v_trimmed = v[:chi,:]

            core_at_index = np.reshape(u_trimmed, newshape=(inp1, out1, left, chi), order='F')

            core_next_index_transposed = np.transpose(s_trimmed * v_trimmed.T, (1,0))
            core_next_index = np.reshape(core_next_index_transposed, newshape=(inp2, out2, chi, right), order='F')
        elif self.__bond_core_index == 0:
            '''
            Core shape decomposition with SVD:

                    |   |             |   |          |       |         |         |         |   |
                    O---O---  --->  [       ]  --->  O---o---O--- ---> O---  o---O--- ---> O---O---
                    |   |             |   |          |       |         |         |         |   |

            (in1, in2, out1, out2, right) -> ((in1, out1), (in1, out2, right))
            (0,     1,    2,    3,     4) -> ((  0,    2), (  1,    3,     4))
            '''
            inp1, inp2, out1, out2, right = self.__bond_core.shape
            newdim = [inp1 * out1, inp2 * out2 * right]

            matrix_core_transposed = np.transpose(self.__bond_core, (0,2,1,3,4))
            matrix_core = np.reshape(matrix_core_transposed, newdim, order='F')
            u, s, v = np.linalg.svd(matrix_core, full_matrices=False)

            u_trimmed = u[:,:chi]
            s_trimmed = s[:chi]
            v_trimmed = v[:chi,:]

            core_at_index = np.reshape(u_trimmed, newshape=(inp1, out1, chi), order='F')

            core_next_index_transposed = np.transpose(s_trimmed * v_trimmed.T, (1,0))
            core_next_index = np.reshape(core_next_index_transposed, newshape=(inp2, out2, chi, right), order='F')
        elif self.__bond_core_index == self.cores_number-2:
            '''
            Core shape decomposition with SVD:

                       |   |              |   |             |       |         |         |           |   |
                    ---O---O  --->   ---[       ]  --->  ---O---o---O ---> ---O---  o---O      ---> O---O---
                       |   |              |   |             |       |         |         |           |   |

            (in1, in2, out1, out2, left) -> ((in1, out1, left), (in1, out2))
            (0,     1,    2,    3,     4) -> ((  0,    2,   4), (  1,    3))
            '''
            inp1, inp2, out1, out2, left = self.__bond_core.shape
            newdim = [inp1 * out1 * left, inp2 * out2]

            matrix_core_transposed = np.transpose(self.__bond_core, (0,2,4,1,3))
            matrix_core = np.reshape(matrix_core_transposed, newdim, order='F')
            u, s, v = np.linalg.svd(matrix_core, full_matrices=False)

            u_trimmed = u[:,:chi]
            s_trimmed = s[:chi]
            v_trimmed = v[:chi,:]
            
            core_at_index = np.reshape(u_trimmed, newshape=(inp1, out1, left, chi), order='F')

            core_next_index_transposed = np.transpose(s_trimmed * v_trimmed.T, (1,0))
            core_next_index = np.reshape(core_next_index_transposed, newshape=(inp2, out2, chi), order='F')

        else:
            raise Exception("Index out of bound")



        if verbose:

            print("Core shape", self.__bond_core.shape)
            print("Matrix unfolding shape : ", newdim)

            print(matrix_core_transposed.shape, newdim)

            print("SVD shapes")
            print(u.shape)
            print(s.shape)
            print(v.shape)
            print("SVD trimmed shapes")
            print(u_trimmed.shape)
            print(s_trimmed.shape)
            print(v_trimmed.shape)

            print("Splitted core shape :", core_at_index.shape)
            print(core_at_index.shape)
            print(core_next_index.shape)

    def __update_bond_core(self, input, index, grad_loss):
        self.__bond_core_update_step = True
        self.__contract_bond_core(index)

        wings = self.__project_wings_MPO(tensor, index)

        # For testing purposes for now
        print(f"Bon dimension at index {index} is {self.__bond_core.shape[4]}")
        self.__SVD_bond_core(chi=self.__bond_core.shape[4], verbose=True)


        # self.__crumble(wings)

        return self.__bond_core - self.learning_rate * grad_loss

    def __crumble(self, projected_wings):
        rank = len(projected_wings.shape)

        print("Rank ", rank, projected_wings.shape)

        if rank != 8 and (self.__bond_core_index != 0 or self.__bond_core_index != self.cores_number-2):
            raise Exception('error while computing the projected input through the wings of the tensor train for a middle bond core')
        elif rank != 3 and (self.__bond_core_index == 0 or self.__bond_core_index == self.cores_number-2):
            raise Exception('error while computing the projected input through the wings of the tensor train for an extreme bond core')

        if rank == 4:
            return np.einsum(projected_wings, np.arange(4), self.__bond_core, np.arange(4))
        elif rank == 3:
            return np.einsum(projected_wings, np.arange(3), self.__bond_core, np.arange(3))
        else:
            raise Exception(f"an error occured with a mismatch of the rank of the projected input through the wings of the tensor train (rank = {rank}")

    def build(self):
        self.cores.append(self.__add_core(name='core_1', type='extreme'))
        for i in range(1, self.cores_number-1):
            self.cores.append(self.__add_core(name = "core_"+str(i), type='middle'))
        self.cores.append(self.__add_core(name='core_'+str(self.cores_number), type='extreme'))

        self.bias = np.zeros(shape=self.output_shape)


    def call(self):
        pass

    def forward(self, input):
        input = np.array(input)
        unfold_input = unfold_shape(input.shape)

        if self.unfold_input_shape != unfold_input:
            exception = f"input of shape {input.shape} cannot be reshaped into {self.input_shape} [{unfold_input} != {self.unfold_input_shape}]"
            raise Exception(exception)
        
        input_tensor = np.reshape(input, newshape=self.input_shape)

        print(input_tensor)

        einsum_structure = []
        input_index = np.arange(self.cores_number)

        einsum_structure.append(input_tensor)
        einsum_structure.append(input_index)

        for idx in range(self.cores_number):
            ipt_index = idx
            opt_index = self.cores_number+idx
            einsum_structure.append(self.cores[idx])
            if idx == 0:
                bnd_index = 2*self.cores_number
                einsum_structure.append([ipt_index, opt_index, bnd_index])
            elif idx == self.cores_number-1:
                bnd_index = 3*self.cores_number-2
                einsum_structure.append([ipt_index, opt_index, bnd_index])
            else:
                bnd_index_1 = 2*self.cores_number+idx-1
                bnd_index_2 = 2*self.cores_number+idx
                einsum_structure.append([ipt_index, opt_index, bnd_index_1, bnd_index_2])

        output_index = np.arange(self.cores_number)+self.cores_number

        einsum_structure.append(output_index)

        print("Structure")
        print(einsum_structure)
        print(len(einsum_structure))

        contraction = np.einsum(*einsum_structure)

        print("Contraction")
        print(contraction)

        result = contraction+self.bias
        print(result)

        return result


    def backward(self, input, grad_output):
        self.__update_bond_core(input, 2, 0)

    def train(self):
        pass


if __name__ == "__main__":

    layer = Dense((2,2,2,2,2,2), (3,3,3,3,3,3), bond_dim=4)
    layer.build()

    print("Cores")
    print(layer.cores)
    print("Bias")
    print(layer.bias)

    tensor = np.arange((2**6)).reshape(((2,2,2,2,2,2)))
    # layer.forward(tensor)

    print('--------- Backpropagation ----------')

    layer.backward(tensor, None)


# HUGE ERROR : I am currently working with MPDO but I implemented a 
# classification backpropagation (thus implementing a MPS)

# I have to that into account the MPO or switch to a more drastic approach consisting
# in creating n times a weight tensor to perform to each the classification approach and backpass
# where n is the number of output (I do not think it is interesting at all)
