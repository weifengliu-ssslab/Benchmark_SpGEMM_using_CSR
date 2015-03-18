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

#ifndef BHSPARSE_H
#define BHSPARSE_H

#include "bhsparse_cuda.h"

class bhsparse
{
public:
    bhsparse();
    int initPlatform(bool *spgemm_platform);
    int initData(int m, int k, int n,
             int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
             int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
             index_type *csrRowPtrC);
    int spgemm();
    int warmup();

    int get_nnzC();
    int get_C(index_type *csrColIndC, value_type *csrValC);

    int freePlatform();
    int free_mem();

private:
    bool *_spgemm_platform;

    bhsparse_cuda   *_bh_sparse_cuda;

    StopWatchInterface *_stage1_timer;
    StopWatchInterface *_stage2_timer;
    StopWatchInterface *_stage3_timer;
    StopWatchInterface *_stage4_timer;

    int spgemm_cuda();

    int statistics();

    int compute_nnzC_Ct_cuda();

    int copy_Ct_to_C_cuda();

    int _m;
    int _k;
    int _n;

    size_t _nnzCt_full;

    // A
    int    _nnzA;
    int   *_h_csrRowPtrA;
    int   *_h_csrColIndA;
    value_type *_h_csrValA;

    // B
    int    _nnzB;
    int   *_h_csrRowPtrB;
    int   *_h_csrColIndB;
    value_type *_h_csrValB;

    // C
    int    _nnzC;
    int   *_h_csrRowPtrC;
    int   *_h_csrColIndC;
    value_type *_h_csrValC;

    // Ct
    //int    _nnzCt;
    int   *_h_csrRowPtrCt;
    //int   *_h_csrColIndCt;
    //value_type *_h_csrValCt;

    // statistics
    int   *_h_counter;
    int   *_h_counter_one;
    int   *_h_counter_sum;
    int   *_h_queue_one;

};

#endif // BHSPARSE_H
