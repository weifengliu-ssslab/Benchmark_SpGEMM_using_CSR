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

__kernel
void compute_nnzCt(__global const int *d_csrRowPtrA,
                   __global const int *d_csrColIndA,
                   __global const int *d_csrRowPtrB,
                   __global int *d_csrRowPtrCt,
                   __local  int *s_csrRowPtrA,
                   const int m)
{
    int global_id  = get_global_id(0);
    int start, stop, index, strideB, row_size_Ct = 0;

    if (global_id < m)
    {
        start = d_csrRowPtrA[global_id];
        stop  = d_csrRowPtrA[global_id + 1];

        for (int i = start; i < stop; i++)
        {
            index = d_csrColIndA[i];
            strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
            row_size_Ct += strideB;
        }

        d_csrRowPtrCt[global_id] = row_size_Ct;
    }

    if (global_id == 0)
        d_csrRowPtrCt[m] = 0;
}
