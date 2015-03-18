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

#include "ref.h"

#include <cusp/multiply.h>
#include <cusp/detail/host/reference/csr.h>

ref::ref()
{
}

template<class I, class T>
void ref::csr_sort_indices(const I n_row,
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


void ref::compData(CSRHost A, CSRHost B, int m, int nnzC, index_type *csrRowPtrC, index_type *csrColIndC, value_type *csrValC)
{
    cout << endl << "Checking correctness ..." << endl;

    COODevice dAcoo = A;
    COODevice dBcoo = B;
    COODevice dCcoo;

    cusp::multiply(dAcoo, dBcoo, dCcoo);

    CSRHost C = dCcoo;

    // check it on CPU, but CUSP v0.4.0's CPU SpGEMM seems incorrect
    //    cusp::coo_matrix<index_type, value_type, cusp::host_memory> Ccoo;
    //    Ccoo = C;
    //    Ccoo.sort_by_row_and_column();
    //    C = Ccoo;

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
