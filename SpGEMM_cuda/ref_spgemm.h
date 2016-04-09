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

#ifndef REF_SPGEMM_H
#define REF_SPGEMM_H

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/detail/host/reference/csr.h>

#include "common.h"

typedef cusp::csr_matrix<index_type,value_type,cusp::host_memory>   CSRHost;
typedef cusp::coo_matrix<index_type,value_type,cusp::device_memory> COODevice;

class ref_spgemm
{
public:
    ref_spgemm();
    template<class I, class T>
    void csr_sort_indices(const I n_row, const I Ap[], I Aj[], T Ax[]);
    void compData(CSRHost A, CSRHost B, int m, int nnzC, index_type *csrRowPtrC, index_type *csrColIndC, value_type *csrValC);
};

ref_spgemm::ref_spgemm()
{
}

template<class I, class T>
void ref_spgemm::csr_sort_indices(const I n_row,
                      const I Ap[],
                            I Aj[],
                            T Ax[])
{
    std::vector< std::pair<I,T> > temp;

    for(I i = 0; i < n_row; i++){
        I row_start = Ap[i];
        I row_end   = Ap[i+1];

        temp.clear();

        for(I jj = row_start; jj < row_end; jj++){
            temp.push_back(std::make_pair(Aj[jj],Ax[jj]));
        }

        std::sort(temp.begin(),temp.end(),kv_pair_less<I,T>);

        for(I jj = row_start, n = 0; jj < row_end; jj++, n++){
            Aj[jj] = temp[n].first;
            Ax[jj] = temp[n].second;
        }
    }
}


void ref_spgemm::compData(CSRHost A, CSRHost B, int m, int nnzC, index_type *csrRowPtrC, index_type *csrColIndC, value_type *csrValC)
{
    cout << endl << "Checking correctness ..." << endl;

    COODevice dAcoo = A;
    COODevice dBcoo = B;
    COODevice dCcoo;

    cusp::multiply(dAcoo, dBcoo, dCcoo);

    CSRHost C = dCcoo;

    csr_sort_indices<index_type, value_type>(m, &C.row_offsets[0], &C.column_indices[0], &C.values[0]);

    // check nnzC
    if (C.num_entries == nnzC)
        cout << "nnzC = " << nnzC << ". PASS!" << endl;
    else
    {
        cout << "nnzC = " << nnzC << ", CUSP's nnzC = " << C.num_entries << ". NO PASS!" << endl;
        return;
    }

    // check csrRowPtrC
    int err_count = 0;
    for (int i = 0; i <= m; i++)
    {
        if (C.row_offsets[i] != csrRowPtrC[i])
            err_count++;
    }
    if (!err_count)
        cout << "RowPtrC PASS!" << endl;
    else
    {
        cout << "RowPtrC NO PASS!" << endl;
        return;
    }

    // check csrColIndC and csrValC
    err_count = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrC[i]; j < csrRowPtrC[i+1]; j++)
        {
            if (C.column_indices[j] != csrColIndC[j] ||
                    fabs((double)C.values[j] - (double)csrValC[j]) > fabs(0.1 * (double)C.values[j]) )
            {
                err_count++;
//                cout << "Row = " << i
//                     << " CUSP: ColIndC = " << C.column_indices[j]
//                     << " ValC = " << C.values[j]
//                     << " BHSPARSE: ColIndC = " << csrColIndC[j]
//                     << " ValC = " << csrValC[j]
//                     << endl;
            }
        }
    }

    if (!err_count)
        cout << "ColIndC/csrValC PASS!" << endl;
    else
        cout << "ColIndC/csrValC NO PASS! #err = " << err_count << endl;
}

#endif // REF_SPGEMM_H
