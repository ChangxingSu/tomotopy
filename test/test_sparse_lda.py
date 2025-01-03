import tomotopy as tp
import pytest

def test_lda_sparse_parameter():
    # Test default parameter
    lda = tp.LDAModel(k=2)
    assert not lda.sparse  # Default should be False
    
    # Test explicit parameter setting
    lda_sparse = tp.LDAModel(k=2, sparse=True)
    assert lda_sparse.sparse
    
    # Test parameter type checking
    with pytest.raises(ValueError):
        tp.LDAModel(k=2, sparse="not_a_bool")

test_lda_sparse_parameter()