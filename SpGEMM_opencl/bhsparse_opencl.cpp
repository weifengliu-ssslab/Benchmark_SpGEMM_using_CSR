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

#include "bhsparse_opencl.h"

bhsparse_opencl::bhsparse_opencl()
{
}

int bhsparse_opencl::initPlatform()
{
    int err = 0;
    _profiling = false;
    int setdevice = 0;

    _warpsize = WARPSIZE_NV;

    // platform
    err = _basicCL.getNumPlatform(&_numPlatforms);
    if(err != CL_SUCCESS) return err;
    cout << "platform number: " << _numPlatforms << ".  ";

    _cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * _numPlatforms);

    err = _basicCL.getPlatformIDs(_cpPlatforms, _numPlatforms);
    if(err != CL_SUCCESS) return err;

    for (unsigned int i = 0; i < _numPlatforms; i++)
    {
        err = _basicCL.getPlatformInfo(_cpPlatforms[i], _platformVendor, _platformVersion);
        if(err != CL_SUCCESS) return err;

        // Gpu device
        err = _basicCL.getNumGpuDevices(_cpPlatforms[i], &_numGpuDevices);

        if (_numGpuDevices > 0)
        {
            _cdGpuDevices = (cl_device_id *)malloc(_numGpuDevices * sizeof(cl_device_id) );

            err |= _basicCL.getGpuDeviceIDs(_cpPlatforms[i], _numGpuDevices, _cdGpuDevices);

            err |= _basicCL.getDeviceInfo(_cdGpuDevices[setdevice], _gpuDeviceName, _gpuDeviceVersion,
                                         &_gpuDeviceComputeUnits, &_gpuDeviceGlobalMem,
                                         &_gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) return err;

            cout << "Platform [" << i <<  "] Vendor: " << _platformVendor << ", Version: " << _platformVersion << endl;
            cout << _numGpuDevices << " Gpu device: "
                 << _gpuDeviceName << " ("
                 << _gpuDeviceComputeUnits << " compute units, "
                 << _gpuDeviceLocalMem / 1024 << " KB local, "
                 << _gpuDeviceGlobalMem / (1024 * 1024) << " MB global, "
                 << _gpuDeviceVersion << ")" << endl;

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = _basicCL.getContext(&_cxGpuContext, _cdGpuDevices, _numGpuDevices);
    if(err != CL_SUCCESS) return err;

    // Gpu commandqueue
    if (_profiling)
        err = _basicCL.getCommandQueueProfilingEnable(&_cqGpuCommandQueue, _cxGpuContext, _cdGpuDevices[setdevice]);
    else
        err = _basicCL.getCommandQueue(&_cqGpuCommandQueue, _cxGpuContext, _cdGpuDevices[setdevice]);
    if(err != CL_SUCCESS) return err;

    _cxLocalContext          = _cxGpuContext;
    _cqLocalCommandQueue     = _cqGpuCommandQueue;

    // get programs
    err  = _basicCL.getProgramFromFile(&_cpWarmup, _cxLocalContext, "warmup_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_computeNnzCt, _cxLocalContext, "SpGEMM_computeNnzCt_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_ESC_0_1,    _cxLocalContext, "SpGEMM_ESC_0_1_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_ESC_2heap,    _cxLocalContext, "SpGEMM_ESC_2heap_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_ESC_bitonic,  _cxLocalContext, "SpGEMM_ESC_bitonic_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_EM,  _cxLocalContext, "SpGEMM_EM_kernels.cl");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getProgramFromFile(&_cpSpGEMM_copyCt2C,     _cxLocalContext, "SpGEMM_copyCt2C_kernels.cl");
    if(err != CL_SUCCESS) return err;

    // get kernels
    err  = _basicCL.getKernel(&_ckWarmup,                               _cpWarmup, "warmup");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckNnzCt,                               _cpSpGEMM_computeNnzCt, "compute_nnzCt");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckESC_0,                               _cpSpGEMM_ESC_0_1,      "ESC_0");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckESC_1,                               _cpSpGEMM_ESC_0_1,      "ESC_1");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckESC_2Heap_NonCoalesced_local,        _cpSpGEMM_ESC_2heap,    "ESC_2heap_noncoalesced_local");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckESC_Bitonic_scan,                    _cpSpGEMM_ESC_bitonic,  "ESC_bitonic_scan");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckEM_mergepath,                        _cpSpGEMM_EM,           "EM_mergepath");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckEM_mergepath_global,                 _cpSpGEMM_EM,           "EM_mergepath_global");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckCopyCt2C_Single,                     _cpSpGEMM_copyCt2C,     "copyCt2C_Single");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckCopyCt2C_Loopless,                   _cpSpGEMM_copyCt2C,     "copyCt2C_Loopless");
    if(err != CL_SUCCESS) return err;
    err  = _basicCL.getKernel(&_ckCopyCt2C_Loop,                       _cpSpGEMM_copyCt2C,     "copyCt2C_Loop");
    if(err != CL_SUCCESS) return err;

    return CL_SUCCESS;
}

int bhsparse_opencl::freePlatform()
{
    int err = 0;

    // free OpenCL event
    err = clReleaseEvent(_ceTimer);    if(err != CL_SUCCESS) return err;

    // free OpenCL kernels
    err = clReleaseKernel(_ckWarmup);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckNnzCt);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckESC_0);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckESC_1);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckESC_2Heap_NonCoalesced_local);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckESC_Bitonic_scan);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckEM_mergepath);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckEM_mergepath_global);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckCopyCt2C_Single);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckCopyCt2C_Loopless);    if(err != CL_SUCCESS) return err;
    err = clReleaseKernel(_ckCopyCt2C_Loop);    if(err != CL_SUCCESS) return err;

    // free OpenCL programs
    err = clReleaseProgram(_cpWarmup);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_computeNnzCt);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_ESC_0_1);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_ESC_2heap);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_copyCt2C);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_ESC_bitonic);    if(err != CL_SUCCESS) return err;
    err = clReleaseProgram(_cpSpGEMM_EM);    if(err != CL_SUCCESS) return err;

    // free OpenCL  devices
    //for (int i = 0; i < _numGpuDevices; i++)
    //{
        //err = clReleaseDevice(_cdGpuDevices[i]);
        //if(err != CL_SUCCESS) return err;
    //}

    // free OpenCL contexts
    //err = clReleaseContext(_cxGpuContext);    if(err != CL_SUCCESS) return err;
    //err = clReleaseContext(_cxLocalContext);    if(err != CL_SUCCESS) return err;

    // free OpenCL command queues
    //err = clReleaseCommandQueue(_cqGpuCommandQueue);    if(err != CL_SUCCESS) return err;
    //err = clReleaseCommandQueue(_cqLocalCommandQueue);    if(err != CL_SUCCESS) return err;

    return err;
}

int bhsparse_opencl::initData(int m, int k, int n,
                              int nnzA, value_type *csrValA, index_type *csrRowPtrA, index_type *csrColIndA,
                              int nnzB, value_type *csrValB, index_type *csrRowPtrB, index_type *csrColIndB,
                              index_type *csrRowPtrC, index_type *csrRowPtrCt, index_type *queue_one,
                              bool use_host_mem)
{
    int err = 0;

    _use_host_mem = use_host_mem;

    _m = m;
    _k = k;
    _n = n;

    _nnzA = nnzA;
    _nnzB = nnzB;
    _nnzC = 0;
    _nnzCt = 0;

    // malloc mem space and copy data from host to device

    // Matrix A
    if (_use_host_mem)
    {
        _d_csrColIndA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, _nnzA  * sizeof(index_type), csrColIndA, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrRowPtrA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (_m+1) * sizeof(index_type), csrRowPtrA, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValA    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, _nnzA  * sizeof(value_type), csrValA, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrColIndA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrRowPtrA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, (_m+1) * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValA    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(value_type), NULL, &err);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrColIndA, CL_TRUE, 0, _nnzA  * sizeof(index_type), csrColIndA, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrRowPtrA, CL_TRUE, 0, (_m+1) * sizeof(index_type), csrRowPtrA, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrValA, CL_TRUE, 0, _nnzA  * sizeof(value_type), csrValA, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    // Matrix B
    if (_use_host_mem)
    {
        _d_csrColIndB = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, _nnzB  * sizeof(index_type), csrColIndB, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrRowPtrB = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (_k+1) * sizeof(index_type), csrRowPtrB, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValB    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, _nnzB  * sizeof(value_type), csrValB, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrColIndB = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzB  * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrRowPtrB = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, (_k+1) * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValB    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzB  * sizeof(value_type), NULL, &err);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrColIndB, CL_TRUE, 0, _nnzB  * sizeof(index_type), csrColIndB, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrRowPtrB, CL_TRUE, 0, (_k+1) * sizeof(index_type), csrRowPtrB, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;

        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrValB, CL_TRUE, 0, _nnzB  * sizeof(value_type), csrValB, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    // Matrix C
    _h_csrRowPtrC = csrRowPtrC;

    if (_use_host_mem)
    {
        _d_csrRowPtrC = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                       (_m+1)  * sizeof(index_type), _h_csrRowPtrC, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrRowPtrC = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, (_m+1)  * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrRowPtrC, CL_TRUE, 0, (_m+1) * sizeof(index_type), _h_csrRowPtrC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    // Matrix Ct
    _h_csrRowPtrCt = csrRowPtrCt;

    if (_use_host_mem)
    {
        _d_csrRowPtrCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        (_m+1)  * sizeof(index_type), _h_csrRowPtrCt, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrRowPtrCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, (_m+1)  * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrRowPtrCt, CL_TRUE, 0, (_m+1) * sizeof(index_type), _h_csrRowPtrCt, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    // statistics - queue_one
    _h_queue_one = queue_one;

    if (_use_host_mem)
    {
        _d_queue_one = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                      TUPLE_QUEUE * _m * sizeof(int), _h_queue_one, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_queue_one = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, TUPLE_QUEUE * _m * sizeof(int), NULL, &err);
        if(err != CL_SUCCESS) return err;
        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_queue_one, CL_TRUE, 0, TUPLE_QUEUE * _m * sizeof(int), _h_queue_one, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    return 0;
}

void bhsparse_opencl::setProfiling(bool profiling)
{
    _profiling = profiling;
}

int bhsparse_opencl::warmup()
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    int num_blocks  = 32 * 1024;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err = clSetKernelArg(_ckWarmup,  0, sizeof(cl_uint) * num_threads * 2, NULL);
    if(err != CL_SUCCESS) { cout << "_ckWarmup arg error = " << err << endl; return err; }

    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckWarmup, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    return err;
}

int bhsparse_opencl::kernel_barrier()
{
    return clFinish(_cqLocalCommandQueue);
}

int bhsparse_opencl::compute_nnzCt()
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    int num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckNnzCt, 0, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
    err |= clSetKernelArg(_ckNnzCt, 1, sizeof(cl_mem), (void*)&_d_csrColIndA);
    err |= clSetKernelArg(_ckNnzCt, 2, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
    err |= clSetKernelArg(_ckNnzCt, 3, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckNnzCt, 4, sizeof(cl_int) * (num_threads + 1), NULL);
    err |= clSetKernelArg(_ckNnzCt, 5, sizeof(cl_int), (void*)&_m);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckNnzCt, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckNnzCt time = " << time << " ms" << endl;
    }

    if (!_use_host_mem)
    {
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_csrRowPtrCt, CL_TRUE, 0, (_m+1) * sizeof(index_type),
                                  _h_csrRowPtrCt, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }
    else
        kernel_barrier();

    return err;
}


int bhsparse_opencl::create_Ct(int nnzCt)
{
    int err = 0;

    if (!_use_host_mem)
    {
        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_queue_one, CL_TRUE, 0, TUPLE_QUEUE * _m * sizeof(int), _h_queue_one, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }
    else
        kernel_barrier();

    _nnzCt = nnzCt;

    // create device mem of Ct
    if (_use_host_mem)
    {
        kernel_barrier();

        _h_csrColIndCt = (index_type *)malloc(_nnzCt * sizeof(index_type));
        _h_csrValCt = (value_type *)malloc(_nnzCt * sizeof(value_type));

        _d_csrColIndCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        _nnzCt * sizeof(index_type), _h_csrColIndCt, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValCt    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        _nnzCt * sizeof(value_type), _h_csrValCt, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrColIndCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, _nnzCt * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValCt    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, _nnzCt * sizeof(value_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
    }

    return err;
}

int bhsparse_opencl::compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckESC_0,  0, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckESC_0,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckESC_0,  2, sizeof(cl_int), (void*)&counter);
    err |= clSetKernelArg(_ckESC_0,  3, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckESC_0, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckESC_0[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckESC_1,  0, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckESC_1,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
    err |= clSetKernelArg(_ckESC_1,  2, sizeof(cl_mem), (void*)&_d_csrColIndA);
    err |= clSetKernelArg(_ckESC_1,  3, sizeof(cl_mem), (void*)&_d_csrValA);
    err |= clSetKernelArg(_ckESC_1,  4, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
    err |= clSetKernelArg(_ckESC_1,  5, sizeof(cl_mem), (void*)&_d_csrColIndB);
    err |= clSetKernelArg(_ckESC_1,  6, sizeof(cl_mem), (void*)&_d_csrValB);
    err |= clSetKernelArg(_ckESC_1,  7, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckESC_1,  8, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckESC_1,  9, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckESC_1, 10, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckESC_1, 11, sizeof(cl_int), (void*)&counter);
    err |= clSetKernelArg(_ckESC_1, 12, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckESC_1, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckESC_1[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::compute_nnzC_Ct_2heap_noncoalesced_local(int num_threads, int num_blocks, int j, int counter, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  0, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  2, sizeof(cl_mem), (void*)&_d_csrColIndA);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  3, sizeof(cl_mem), (void*)&_d_csrValA);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  4, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  5, sizeof(cl_mem), (void*)&_d_csrColIndB);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  6, sizeof(cl_mem), (void*)&_d_csrValB);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  7, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  8, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local,  9, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local, 10, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local, 11, sizeof(cl_int)   * j * num_threads, NULL);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local, 12, sizeof(cl_double) * j * num_threads, NULL);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local, 13, sizeof(cl_int), (void*)&counter);
    err |= clSetKernelArg(_ckESC_2Heap_NonCoalesced_local, 14, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckESC_2Heap_NonCoalesced_local, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckESC_2Heap_NonCoalesced_local[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::compute_nnzC_Ct_bitonic_scan(int num_threads, int num_blocks, int j, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;
    int buffer_size = 2 * num_threads;

    err  = clSetKernelArg(_ckESC_Bitonic_scan,  0, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  2, sizeof(cl_mem), (void*)&_d_csrColIndA);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  3, sizeof(cl_mem), (void*)&_d_csrValA);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  4, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  5, sizeof(cl_mem), (void*)&_d_csrColIndB);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  6, sizeof(cl_mem), (void*)&_d_csrValB);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  7, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  8, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckESC_Bitonic_scan,  9, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 10, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 11, sizeof(cl_int)   * buffer_size, NULL);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 12, sizeof(cl_double) * buffer_size, NULL);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 13, sizeof(cl_short) * (buffer_size + 1), NULL);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 14, sizeof(cl_int), (void*)&position);
    err |= clSetKernelArg(_ckESC_Bitonic_scan, 15, sizeof(cl_int), (void*)&_n);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckESC_Bitonic_scan, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckESC_Bitonic_scan[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size,
                                               int position, int *count_next, int mergepath_location)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    if (mergepath_location == MERGEPATH_LOCAL)
    {
        err  = clSetKernelArg(_ckEM_mergepath,  0, sizeof(cl_mem), (void*)&_d_queue_one);
        err |= clSetKernelArg(_ckEM_mergepath,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
        err |= clSetKernelArg(_ckEM_mergepath,  2, sizeof(cl_mem), (void*)&_d_csrColIndA);
        err |= clSetKernelArg(_ckEM_mergepath,  3, sizeof(cl_mem), (void*)&_d_csrValA);
        err |= clSetKernelArg(_ckEM_mergepath,  4, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
        err |= clSetKernelArg(_ckEM_mergepath,  5, sizeof(cl_mem), (void*)&_d_csrColIndB);
        err |= clSetKernelArg(_ckEM_mergepath,  6, sizeof(cl_mem), (void*)&_d_csrValB);
        err |= clSetKernelArg(_ckEM_mergepath,  7, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
        err |= clSetKernelArg(_ckEM_mergepath,  8, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
        err |= clSetKernelArg(_ckEM_mergepath,  9, sizeof(cl_mem), (void*)&_d_csrColIndCt);
        err |= clSetKernelArg(_ckEM_mergepath, 10, sizeof(cl_mem), (void*)&_d_csrValCt);
        err |= clSetKernelArg(_ckEM_mergepath, 11, sizeof(cl_int)   * (mergebuffer_size + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath, 12, sizeof(cl_double) * (mergebuffer_size + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath, 13, sizeof(cl_short) * (num_threads + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath, 14, sizeof(cl_int), (void*)&position);
        err |= clSetKernelArg(_ckEM_mergepath, 15, sizeof(cl_int), (void*)&mergebuffer_size);
        if(err != CL_SUCCESS) { cout << "mergepath arg error = " << err << endl; return err; }

        // run kernel
        err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckEM_mergepath, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }
    }
    else if (mergepath_location == MERGEPATH_GLOBAL)
    {
        int mergebuffer_size_local = 2560;

        //cout << "-------------------doing global" << endl;
        err  = clSetKernelArg(_ckEM_mergepath_global,  0, sizeof(cl_mem), (void*)&_d_queue_one);
        err |= clSetKernelArg(_ckEM_mergepath_global,  1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
        err |= clSetKernelArg(_ckEM_mergepath_global,  2, sizeof(cl_mem), (void*)&_d_csrColIndA);
        err |= clSetKernelArg(_ckEM_mergepath_global,  3, sizeof(cl_mem), (void*)&_d_csrValA);
        err |= clSetKernelArg(_ckEM_mergepath_global,  4, sizeof(cl_mem), (void*)&_d_csrRowPtrB);
        err |= clSetKernelArg(_ckEM_mergepath_global,  5, sizeof(cl_mem), (void*)&_d_csrColIndB);
        err |= clSetKernelArg(_ckEM_mergepath_global,  6, sizeof(cl_mem), (void*)&_d_csrValB);
        err |= clSetKernelArg(_ckEM_mergepath_global,  7, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
        err |= clSetKernelArg(_ckEM_mergepath_global,  8, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
        err |= clSetKernelArg(_ckEM_mergepath_global,  9, sizeof(cl_mem), (void*)&_d_csrColIndCt);
        err |= clSetKernelArg(_ckEM_mergepath_global, 10, sizeof(cl_mem), (void*)&_d_csrValCt);
        err |= clSetKernelArg(_ckEM_mergepath_global, 11, sizeof(cl_int)   * (mergebuffer_size_local + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath_global, 12, sizeof(cl_double) * (mergebuffer_size_local + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath_global, 13, sizeof(cl_short) * (num_threads + 1), NULL);
        err |= clSetKernelArg(_ckEM_mergepath_global, 14, sizeof(cl_int), (void*)&position);
        err |= clSetKernelArg(_ckEM_mergepath_global, 15, sizeof(cl_int), (void*)&mergebuffer_size_local);
        if(err != CL_SUCCESS) { cout << "mergepath arg error = " << err << endl; return err; }

        // run kernel
        err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckEM_mergepath_global, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }
    }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckEM_mergepath[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    // load d_queue back, check if there is still any row needs next level merge,
    if (!_use_host_mem)
    {
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_queue_one, CL_TRUE, TUPLE_QUEUE * position * sizeof(int), TUPLE_QUEUE * num_blocks * sizeof(int),
                                  &_h_queue_one[TUPLE_QUEUE * position], 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }
    else
        kernel_barrier();

    int temp_queue [6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;
    for (int i = position; i < position + num_blocks; i++)
    {
        // if yes, (1)malloc device mem, (2)upgrade mem address on pos1 and (3)use pos5 as last mem address
        if (_h_queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = _h_queue_one[TUPLE_QUEUE * i]; // row id
            if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
            {
                //temp_queue[1] = _nnzCt + counter * mergebuffer_size * 2; // new start address
                int accum = 0;
                switch (mergebuffer_size)
                {
                case 256:
                    accum = 512;
                    break;
                case 512:
                    accum = 1024;
                    break;
                case 1024:
                    accum = 2048;
                    break;
                case 2048:
                    accum = 2560;
                    break;
                case 2560:
                    accum = 2560 * 2;
                    break;
                }

                temp_queue[1] = _nnzCt + counter * accum; // new start address
            }
            else if (mergepath_location == MERGEPATH_GLOBAL)
                temp_queue[1] = _nnzCt + counter * (mergebuffer_size + 2560); // new start address
            temp_queue[2] = _h_queue_one[TUPLE_QUEUE * i + 2]; // merged size
            temp_queue[3] = _h_queue_one[TUPLE_QUEUE * i + 3]; // i
            temp_queue[4] = _h_queue_one[TUPLE_QUEUE * i + 4]; // k
            temp_queue[5] = _h_queue_one[TUPLE_QUEUE * i + 1]; // old start address

            _h_queue_one[TUPLE_QUEUE * i]     = _h_queue_one[TUPLE_QUEUE * (position + counter)];     // row id
            _h_queue_one[TUPLE_QUEUE * i + 1] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 1]; // new start address
            _h_queue_one[TUPLE_QUEUE * i + 2] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 2]; // merged size
            _h_queue_one[TUPLE_QUEUE * i + 3] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 3]; // i
            _h_queue_one[TUPLE_QUEUE * i + 4] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 4]; // k
            _h_queue_one[TUPLE_QUEUE * i + 5] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 5]; // old start address

            _h_queue_one[TUPLE_QUEUE * (position + counter)]     = temp_queue[0]; // row id
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 1] = temp_queue[1]; // new start address
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 2] = temp_queue[2]; // merged size
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 3] = temp_queue[3]; // i
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 4] = temp_queue[4]; // k
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 5] = temp_queue[5]; // old start address

            counter++;
            temp_num += _h_queue_one[TUPLE_QUEUE * i + 2];
            //cout << counter << " ) _h_queue_one[TUPLE_QUEUE * i + 2] = " << _h_queue_one[TUPLE_QUEUE * i + 2] << endl;
        }
    }

    if (counter > 0)
    {
        int nnzCt_new;
        if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
        {
            //nnzCt_new = _nnzCt + counter * mergebuffer_size * 2; // new start address
            int accum = 0;
            switch (mergebuffer_size)
            {
            case 256:
                accum = 512;
                break;
            case 512:
                accum = 1024;
                break;
            case 1024:
                accum = 2048;
                break;
            case 2048:
                accum = 2560;
                break;
            case 2560:
                accum = 2560 * 2;
                break;
            }

            nnzCt_new = _nnzCt + counter * accum;
        }
        else if (mergepath_location == MERGEPATH_GLOBAL)
            nnzCt_new = _nnzCt + counter * (mergebuffer_size + 2560);
        cout << "nnzCt_new = " << nnzCt_new << endl;

        // malloc new device memory
        // copy last device mem to current one, device to device copy
        // free last device mem
        if (_use_host_mem)
        {
            kernel_barrier();

            index_type *h_csrColIndCt_new = (index_type *)realloc(_h_csrColIndCt, nnzCt_new * sizeof(index_type));
            if (h_csrColIndCt_new != NULL)
            {
                _h_csrColIndCt = h_csrColIndCt_new;
                _d_csrColIndCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                nnzCt_new * sizeof(index_type), _h_csrColIndCt, &err);
                if(err != CL_SUCCESS) return err;
            }
            else
            {
                cout << "ColIndCt re-allocation error." << endl;
                return err;
            }

            value_type *h_csrValCt_new = (value_type *)realloc(_h_csrValCt, nnzCt_new * sizeof(value_type));
            if (h_csrValCt_new != NULL)
            {
                _h_csrValCt = h_csrValCt_new;
                _d_csrValCt    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                nnzCt_new * sizeof(value_type), _h_csrValCt, &err);
                if(err != CL_SUCCESS) return err;
            }
            else
            {
                cout << "ValCt re-allocation error." << endl;
                return err;
            }
        }
//        if (_use_host_mem) // use for benchmarking phase 2 (malloc + memcpy)
//        {
//            kernel_barrier();

//            index_type *h_csrColIndCt_new = (index_type *)malloc(nnzCt_new * sizeof(index_type));
//            memcpy(h_csrColIndCt_new, _h_csrColIndCt, _nnzCt * sizeof(index_type));
//            free(_h_csrColIndCt);
//            _h_csrColIndCt = h_csrColIndCt_new;
//            _d_csrColIndCt = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
//                                            nnzCt_new * sizeof(index_type), _h_csrColIndCt, &err);
//            if(err != CL_SUCCESS) return err;

//            value_type *h_csrValCt_new = (value_type *)malloc(nnzCt_new * sizeof(value_type));
//            memcpy(h_csrValCt_new, _h_csrValCt, _nnzCt * sizeof(value_type));
//            free(_h_csrValCt);
//            _h_csrValCt = h_csrValCt_new;
//            _d_csrValCt    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
//                                            nnzCt_new * sizeof(value_type), _h_csrValCt, &err);
//            if(err != CL_SUCCESS) return err;
//        }
        else
        {
            cl_mem d_csrColIndCt_new;
            d_csrColIndCt_new = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, nnzCt_new * sizeof(index_type), NULL, &err);

            err = clEnqueueCopyBuffer(_cqLocalCommandQueue, _d_csrColIndCt, d_csrColIndCt_new, 0, 0, _nnzCt * sizeof(index_type), 0, NULL, NULL);
            if(err != CL_SUCCESS)
            {
                index_type *h_csrColIndCt = (index_type *)malloc(_nnzCt  * sizeof(index_type));
                // copy last device mem to a temp space on host
                err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                          _d_csrColIndCt, CL_TRUE, 0, _nnzCt  * sizeof(index_type),
                                           h_csrColIndCt, 0, NULL, NULL);
                if(err != CL_SUCCESS) return err;

                // free last device mem
                if(_d_csrColIndCt) err = clReleaseMemObject(_d_csrColIndCt);
                if(err != CL_SUCCESS) return err;

                // copy data in the temp space on host to device
                err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                           d_csrColIndCt_new, CL_TRUE, 0, _nnzCt  * sizeof(index_type),
                                           h_csrColIndCt, 0, NULL, NULL);
                if(err != CL_SUCCESS) return err;

                free(h_csrColIndCt);
            }
            else
            {
                if(_d_csrColIndCt) err = clReleaseMemObject(_d_csrColIndCt);
                if(err != CL_SUCCESS) return err;
            }

            _d_csrColIndCt = d_csrColIndCt_new;

            cl_mem d_csrValCt_new;
            d_csrValCt_new = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, nnzCt_new * sizeof(value_type), NULL, &err);

            err = clEnqueueCopyBuffer(_cqLocalCommandQueue, _d_csrValCt, d_csrValCt_new, 0, 0, _nnzCt * sizeof(value_type), 0, NULL, NULL);
            if(err != CL_SUCCESS)
            {
                value_type *h_csrValCt = (value_type *)malloc(_nnzCt  * sizeof(value_type));
                // copy last device mem to a temp space on host
                err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                          _d_csrValCt, CL_TRUE, 0, _nnzCt  * sizeof(value_type),
                                           h_csrValCt, 0, NULL, NULL);
                if(err != CL_SUCCESS) return err;

                // free last device mem
                if(_d_csrValCt) err = clReleaseMemObject(_d_csrValCt);
                if(err != CL_SUCCESS) return err;

                // copy data in the temp space on host to device
                err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                           d_csrValCt_new, CL_TRUE, 0, _nnzCt  * sizeof(value_type),
                                           h_csrValCt, 0, NULL, NULL);
                if(err != CL_SUCCESS) return err;

                free(h_csrValCt);
            }
            else
            {
                if(_d_csrValCt) err = clReleaseMemObject(_d_csrValCt);
                if(err != CL_SUCCESS) return err;
            }

            _d_csrValCt    = d_csrValCt_new;
        }

        // rewrite d_queue
        if (!_use_host_mem)
        {
            err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                       _d_queue_one, CL_TRUE, TUPLE_QUEUE * position * sizeof(int), TUPLE_QUEUE * num_blocks * sizeof(int),
                                       &_h_queue_one[TUPLE_QUEUE * position], 0, NULL, NULL);
            if(err != CL_SUCCESS) return err;
        }
        else
            kernel_barrier();

        _nnzCt = nnzCt_new;
    }

    *count_next = counter;

    return err;
}

int bhsparse_opencl::create_C()
{
    int err = 0;

    if (!_use_host_mem)
    {
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_csrRowPtrC, CL_TRUE, 0, (_m + 1) * sizeof(index_type),
                                  _h_csrRowPtrC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }
    else
        kernel_barrier();

    int old_val, new_val;
    old_val = _h_csrRowPtrC[0];
    _h_csrRowPtrC[0] = 0;
    for (int i = 1; i <= _m; i++)
    {
        new_val = _h_csrRowPtrC[i];
        _h_csrRowPtrC[i] = old_val + _h_csrRowPtrC[i-1];
        old_val = new_val;
    }

    _nnzC = _h_csrRowPtrC[_m];
    cout << "nnzC = " << _nnzC << endl;

    // create device mem of C
    if (_use_host_mem)
    {
        kernel_barrier();

        _h_csrColIndC = (index_type *)malloc(_nnzC * sizeof(index_type));
        _h_csrValC = (value_type *)malloc(_nnzC * sizeof(value_type));

        _d_csrColIndC = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        _nnzC * sizeof(index_type), _h_csrColIndC, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValC    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        _nnzC * sizeof(value_type), _h_csrValC, &err);
        if(err != CL_SUCCESS) return err;
    }
    else
    {
        _d_csrColIndC = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, _nnzC * sizeof(index_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
        _d_csrValC    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, _nnzC * sizeof(value_type), NULL, &err);
        if(err != CL_SUCCESS) return err;
    }

    if (!_use_host_mem)
    {
        err = clEnqueueWriteBuffer(_cqLocalCommandQueue,
                                   _d_csrRowPtrC, CL_TRUE, 0, (_m + 1) * sizeof(index_type),
                                   _h_csrRowPtrC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }
    else
        kernel_barrier();

    return err;
}

int bhsparse_opencl::copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position)
{
    int err = 0;

    int j = 1;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckCopyCt2C_Single,  0, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  1, sizeof(cl_mem), (void*)&_d_csrColIndC);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  2, sizeof(cl_mem), (void*)&_d_csrValC);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  3, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  4, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  5, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  6, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  7, sizeof(cl_int), (void*)&local_size);
    err |= clSetKernelArg(_ckCopyCt2C_Single,  8, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckCopyCt2C_Single, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "single kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckCopyCt2C_Single[ " << j <<  " ] time   = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckCopyCt2C_Loopless,  0, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  1, sizeof(cl_mem), (void*)&_d_csrColIndC);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  2, sizeof(cl_mem), (void*)&_d_csrValC);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  3, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  4, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  5, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  6, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckCopyCt2C_Loopless,  7, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckCopyCt2C_Loopless, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "loopless event error = " << err << ", j = " << j << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckCopyCt2C_Loopless[ " << j <<  " ] time = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position)
{
    int err = 0;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    double time;

    err  = clSetKernelArg(_ckCopyCt2C_Loop,  0, sizeof(cl_mem), (void*)&_d_csrRowPtrC);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  1, sizeof(cl_mem), (void*)&_d_csrColIndC);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  2, sizeof(cl_mem), (void*)&_d_csrValC);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  3, sizeof(cl_mem), (void*)&_d_csrRowPtrCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  4, sizeof(cl_mem), (void*)&_d_csrColIndCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  5, sizeof(cl_mem), (void*)&_d_csrValCt);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  6, sizeof(cl_mem), (void*)&_d_queue_one);
    err |= clSetKernelArg(_ckCopyCt2C_Loop,  7, sizeof(cl_int), (void*)&position);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckCopyCt2C_Loop, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &_ceTimer);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    if (_profiling)
    {
        err = clWaitForEvents(1, &_ceTimer);
        if(err != CL_SUCCESS) { cout << "loop event error = " << err << endl; return err; }

        _basicCL.getEventTimer(_ceTimer, &_queuedTime, &_submitTime, &_startTime, &_endTime);
        time = double(_endTime - _submitTime) / 1000000.0;

        cout << "_ckCopyCt2C_Loop[ " << j <<  " ] time     = " << time << " ms" << endl;
    }

    return err;
}

int bhsparse_opencl::get_nnzC()
{
    return _nnzC;
}

int bhsparse_opencl::get_C(index_type *csrColIndC, value_type *csrValC)
{
    int err = 0;

    if (_use_host_mem)
    {
        kernel_barrier();
        memcpy(csrColIndC, _h_csrColIndC, _nnzC * sizeof(index_type));
        memcpy(csrValC, _h_csrValC, _nnzC * sizeof(value_type));
    }
    else
    {
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_csrColIndC, CL_TRUE, 0, _nnzC * sizeof(index_type),
                                  csrColIndC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_csrRowPtrC, CL_TRUE, 0, (_m + 1) * sizeof(index_type),
                                  _h_csrRowPtrC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
        err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                                  _d_csrValC, CL_TRUE, 0, _nnzC * sizeof(value_type),
                                  csrValC, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
    }

    return err;
}

int bhsparse_opencl::free_mem()
{
    int err = 0;

    // A
    if (_use_host_mem)
    {
        kernel_barrier();
        if(_d_csrValA) err = clReleaseMemObject(_d_csrValA); if(err != CL_SUCCESS) return err;
        if(_d_csrRowPtrA) err = clReleaseMemObject(_d_csrRowPtrA); if(err != CL_SUCCESS) return err;
        if(_d_csrColIndA) err = clReleaseMemObject(_d_csrColIndA); if(err != CL_SUCCESS) return err;
    }

    // B
    if (_use_host_mem)
    {
        kernel_barrier();
        if(_d_csrValB) err = clReleaseMemObject(_d_csrValB); if(err != CL_SUCCESS) return err;
        if(_d_csrRowPtrB) err = clReleaseMemObject(_d_csrRowPtrB); if(err != CL_SUCCESS) return err;
        if(_d_csrColIndB) err = clReleaseMemObject(_d_csrColIndB); if(err != CL_SUCCESS) return err;
    }

    // C
    if (_use_host_mem)
    {
        kernel_barrier();
        if(_d_csrRowPtrC) err = clReleaseMemObject(_d_csrRowPtrC); if(err != CL_SUCCESS) return err;
        if(_d_csrColIndC) err = clReleaseMemObject(_d_csrColIndC); if(err != CL_SUCCESS) return err;
        if(_d_csrValC) err = clReleaseMemObject(_d_csrValC); if(err != CL_SUCCESS) return err;
    }
    else
    {
        free(_h_csrColIndC);
        free(_h_csrValC);
    }

    // Ct

    if (_use_host_mem)
    {
        kernel_barrier();
        if(_d_csrRowPtrCt) err = clReleaseMemObject(_d_csrRowPtrCt); if(err != CL_SUCCESS) return err;
        if(_d_csrValCt) err = clReleaseMemObject(_d_csrValCt); if(err != CL_SUCCESS) return err;
        if(_d_csrColIndCt) err = clReleaseMemObject(_d_csrColIndCt); if(err != CL_SUCCESS) return err;
    }
    else
    {
        free(_h_csrColIndCt);
        free(_h_csrValCt);
    }

    // QUEUE_ONEs
    if (_use_host_mem)
    {
        kernel_barrier();
        if(_d_queue_one) err = clReleaseMemObject(_d_queue_one); if(err != CL_SUCCESS) return err;
    }

    return err;
}
