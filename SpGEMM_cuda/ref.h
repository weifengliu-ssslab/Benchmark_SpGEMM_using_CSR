//////////////////////////////////////////////////////////////////////////
// < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
//
// < See paper:
// Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
// Matrix-Matrix Multiplication for Irregular Data," Parallel and
// Distributed Processing Symposium, 2014 IEEE 28th International
// (IPDPS '14), pp.370-381, 19-23 May 2014
// for details. >
//////////////////////////////////////////////////////////////////////////

#ifndef REF_H
#define REF_H

#include <cusp/csr_matrix.h>

#include "common.h"

typedef cusp::csr_matrix<index_type,value_type,cusp::host_memory>   CSRHost;
typedef cusp::coo_matrix<index_type,value_type,cusp::device_memory> COODevice;

class ref
{
public:
    ref();
    template<class I, class T>
    void csr_sort_indices(const I n_row, const I Ap[], I Aj[], T Ax[]);
    void compData(CSRHost A, CSRHost B, int m, int nnzC, index_type *csrRowPtrC, index_type *csrColIndC, value_type *csrValC);
};

#endif // REF_H
