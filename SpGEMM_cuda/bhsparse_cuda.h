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

#ifndef BHSPARSE_CUDA_H
#define BHSPARSE_CUDA_H

#include "common.h"

class bhsparse_cuda
{
public:
    bhsparse_cuda();
    int initPlatform();
    int initData(int m, int k, int n,
             int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
             int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
             index_type *csrRowPtrC, index_type *csrRowPtrCt, index_type *queue_one);

    void setProfiling(bool profiling);
    int kernel_barrier();
    int warmup();
    int freePlatform();
    int free_mem();

    int compute_nnzCt();
    int create_Ct(int nnzCt);
    int create_C();

    int compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_2heap_noncoalesced(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_bitonic(int num_threads, int num_blocks, int j, int position);
    int compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location);


    int copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position);
    int copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position);
    int copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position);

    int get_nnzC();
    int get_C(index_type *csrColIndC, value_type *csrValC);

private:
    bool _profiling;

    int _num_smxs;
    int _max_blocks_per_smx;

    int _m;
    int _k;
    int _n;

    // A
    int _nnzA;
    value_type *_d_csrValA;
    index_type *_d_csrRowPtrA;
    index_type *_d_csrColIndA;

    // B
    int _nnzB;
    value_type *_d_csrValB;
    index_type *_d_csrRowPtrB;
    index_type *_d_csrColIndB;

    // C
    int _nnzC;
    index_type *_h_csrRowPtrC;
    index_type *_d_csrRowPtrC;
    index_type *_d_csrColIndC;
    value_type *_d_csrValC;

    // Ct
    int _nnzCt;
    value_type *_d_csrValCt;
    index_type *_d_csrColIndCt;
    index_type *_d_csrRowPtrCt;
    index_type *_h_csrRowPtrCt;

    // QUEUE_ONEs
    index_type *_h_queue_one;
    index_type *_d_queue_one;
};

#endif // BHSPARSE_CUDA_H
