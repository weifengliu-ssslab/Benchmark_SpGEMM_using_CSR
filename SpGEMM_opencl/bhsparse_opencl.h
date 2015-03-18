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

#ifndef BHSPARSE_OPENCL_H
#define BHSPARSE_OPENCL_H

#include "common.h"
#include "basiccl.h"

class bhsparse_opencl
{
public:
    bhsparse_opencl();
    int initPlatform();
    int initData(int m, int k, int n,
             int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
             int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
             index_type *csrRowPtrC, index_type *csrRowPtrCt, index_type *queue_one, bool use_host_mem);

    void setProfiling(bool profiling);
    int kernel_barrier();
    int warmup();
    int freePlatform();
    int free_mem();

    int compute_nnzCt();
    int compute_nnzC();

    int create_Ct(int nnzCt);
    int create_C();

    int compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_2heap_noncoalesced_local(int num_threads, int num_blocks, int j, int counter, int position);
    int compute_nnzC_Ct_bitonic_scan(int num_threads, int num_blocks, int j, int position);
    int compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size,
                                  int position, int *count_next, int mergepath_location);

    int copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position);
    int copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position);
    int copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position);

    int get_nnzC();
    int get_C(index_type *csrColIndC, value_type *csrValC);

private:

    bool _use_host_mem;
    bool _profiling;

    int _m;
    int _k;
    int _n;

    // basic OpenCL variables
    BasicCL _basicCL;

    char _platformVendor[CL_STRING_LENGTH];
    char _platformVersion[CL_STRING_LENGTH];

    char _gpuDeviceName[CL_STRING_LENGTH];
    char _gpuDeviceVersion[CL_STRING_LENGTH];
    int  _gpuDeviceComputeUnits;
    int  _gpuDeviceGlobalMem;
    int  _gpuDeviceLocalMem;
    int  _localDeviceComputeUnits;

    int          _warpsize;

    cl_uint             _numPlatforms;           // OpenCL platform
    cl_platform_id*     _cpPlatforms;

    cl_uint             _numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       _cdGpuDevices;

    cl_context          _cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    _cqGpuCommandQueue;      // OpenCL Gpu command queues

    cl_context          _cxLocalContext;         // OpenCL Local context
    cl_command_queue    _cqLocalCommandQueue;    // OpenCL Local command queues

    cl_program          _cpWarmup;               // OpenCL Gpu program
    cl_program          _cpSpGEMM_computeNnzCt;    // OpenCL Gpu program
    cl_program          _cpSpGEMM_ESC_0_1;         // OpenCL Gpu program
    cl_program          _cpSpGEMM_ESC_2heap;       // OpenCL Gpu program
    cl_program          _cpSpGEMM_copyCt2C;        // OpenCL Gpu program
    cl_program          _cpSpGEMM_ESC_bitonic;     // OpenCL Gpu program
    cl_program          _cpSpGEMM_EM;              // OpenCL Gpu program

    cl_kernel           _ckWarmup;                           // OpenCL Gpu kernel
    cl_kernel           _ckNnzCt;                            // OpenCL Gpu kernel
    cl_kernel           _ckESC_0;                            // OpenCL Gpu kernel
    cl_kernel           _ckESC_1;                            // OpenCL Gpu kernel
    cl_kernel           _ckESC_2Heap_NonCoalesced_local;     // OpenCL Gpu kernel
    cl_kernel           _ckESC_Bitonic_scan;                 // OpenCL Gpu kernel
    cl_kernel           _ckEM_mergepath;                     // OpenCL Gpu kernel
    cl_kernel           _ckEM_mergepath_global;              // OpenCL Gpu kernel
    cl_kernel           _ckCopyCt2C_Single;                  // OpenCL Gpu kernel
    cl_kernel           _ckCopyCt2C_Loopless;                // OpenCL Gpu kernel
    cl_kernel           _ckCopyCt2C_Loop;                    // OpenCL Gpu kernel

    cl_event            _ceTimer;                            // OpenCL event

    cl_ulong            _queuedTime;
    cl_ulong            _submitTime;
    cl_ulong            _startTime;
    cl_ulong            _endTime;

    // A
    int _nnzA;
    cl_mem _d_csrValA;
    cl_mem _d_csrRowPtrA;
    cl_mem _d_csrColIndA;

    // B
    int _nnzB;
    cl_mem _d_csrValB;
    cl_mem _d_csrRowPtrB;
    cl_mem _d_csrColIndB;

    // C
    int _nnzC;
    index_type *_h_csrRowPtrC;
    index_type *_h_csrColIndC;
    value_type *_h_csrValC;
    cl_mem      _d_csrRowPtrC;
    cl_mem      _d_csrColIndC;
    cl_mem      _d_csrValC;

    int _nnzCt;
    index_type *_h_csrRowPtrCt;
    index_type *_h_csrColIndCt;
    value_type *_h_csrValCt;
    cl_mem      _d_csrValCt;
    cl_mem      _d_csrColIndCt;
    cl_mem      _d_csrRowPtrCt;

    // QUEUE_ONEs
    index_type *_h_queue_one;
    cl_mem      _d_queue_one;
};

#endif // BHSPARSE_OPENCL_H
