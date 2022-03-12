from syngular.tensor import MatrixProductOperator
from syngular.tensor import MatrixProductState

from typing import Union

from opt_einsum import contract

def mul(op1: Union[MatrixProductOperator, MatrixProductState], op2: Union[MatrixProductOperator, MatrixProductState], mode="standard"):
    n = op1.sites_number
    m = op2.sites_number

    min_bond = min(min(op1.bond_shape), min(op2.bond_shape))
    max_bond = max(max(op1.bond_shape), max(op2.bond_shape))

    if n != m:
        raise Exception("both operator do not have the same number of sites")
    
    '''
    Standard mode:
        - basic contractions of physicial legs
        - compression algorithm afterwards

    > MPO / MPS 
    > MPS / MPS
    > MPO / MPO
    '''
    if mode == "standard":
        sites = []

        if  isinstance(op1, MatrixProductState) and isinstance(op2, MatrixProductOperator):
            if not op1.decomposed or not op2.decomposed:
                raise Exception("Operators and States must be decomposed")
            
            for idx in range(n):
                site = contract(op1.sites[idx], [1,2,3], op2.sites[idx], [4,2,5,6], [1,4,5,3,6])

                site = site.reshape((
                    op2.shape[idx][0]*op1.shape[idx][0],
                    op2.shape[idx][2],
                    op2.shape[idx][3]*op1.shape[idx][2]
                ))
                sites.append(site)
            return MatrixProductState.from_sites(sites) >> min_bond
        elif isinstance(op1, MatrixProductOperator) and isinstance(op2, MatrixProductState):
            if not op1.decomposed or not op2.decomposed:
                raise Exception("Operators and States must be decomposed")
            
            for idx in range(n):
                site = contract(op2.sites[idx], [1,2,3], op1.sites[idx], [4,2,5,6], [1,4,5,3,6])

                site = site.reshape((
                    op1.shape[idx][0]*op2.shape[idx][0],
                    op1.shape[idx][2],
                    op1.shape[idx][3]*op2.shape[idx][2]
                ))
                sites.append(site)
            return MatrixProductState.from_sites(sites) >> min_bond
        elif isinstance(op1, MatrixProductState) and isinstance(op2, MatrixProductState):
            return op1 | op2
        elif isinstance(op1, MatrixProductOperator) and isinstance(op2, MatrixProductOperator):
            for idx in range(n):
                site = contract(op2.sites[idx], [1,2,3,4], op1.sites[idx], [5,3,6,7], [1,5,2,6,4,7])

                site = site.reshape((
                    op1.shape[idx][0]*op2.shape[idx][0],
                    op1.shape[idx][1],
                    op2.shape[idx][2],
                    op1.shape[idx][3]*op2.shape[idx][3]
                ))
                sites.append(site)
            return MatrixProductOperator.from_sites(sites) >> min_bond
        else:
            raise Exception("`syn.mul` should be provided MatrixProductState or MatrixProductOperator objects only")
        

    '''
    Variational mode:
        - compute left and right blocks
        - update random state
        - sweep
    '''
    if mode == "variational":
        pass
    
    '''
    Standard mode:
        - basic contractions of physicial legs
        - compression algorithm afterwards
    '''
    if mode == "optimized":
        pass
    
    '''
    Standard mode:
        - basic contractions of physicial legs
        - compression algorithm afterwards
    '''
    if mode == "fitup":
        pass

