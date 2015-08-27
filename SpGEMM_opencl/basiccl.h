/* ************************************************************************
* The MIT License (MIT)
* Copyright 2014-2015 weifengliu
 
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
 
*  The above copyright notice and this permission notice shall be included in
*  all copies or substantial portions of the Software.
 
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*  THE SOFTWARE.
* ************************************************************************ */
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

#ifndef BASICCL_H
#define BASICCL_H

#include "common.h"

#define CL_STRING_LENGTH 128

class BasicCL
{
public:
    BasicCL();

    int getNumPlatform(cl_uint *numPlatforms);
    int getPlatformIDs(cl_platform_id *platforms, cl_uint numPlatforms);
    int getPlatformInfo(cl_platform_id platform, char *platformVendor, char *platformVersion);

    int getNumCpuDevices(cl_platform_id platform, cl_uint *numCpuDevices);
    int getNumGpuDevices(cl_platform_id platform, cl_uint *numGpuDevices);
    int getCpuDeviceIDs(cl_platform_id platform, cl_uint numCpuDevices, cl_device_id *cpuDevices);
    int getGpuDeviceIDs(cl_platform_id platform, cl_uint numGpuDevices, cl_device_id *gpuDevices);
    int getDeviceInfo(cl_device_id device, char *deviceName, char *deviceVersion,
                      int *deviceComputeUnits, int *deviceGlobalMem,
                      int *deviceLocalMem, int *maxSubDevices);

    int getContext(cl_context *context, cl_device_id *devices, cl_uint numDevices);

    int getCommandQueue(cl_command_queue *commandQueue, cl_context context, cl_device_id device);
    int getCommandQueueProfilingEnable(cl_command_queue *commandQueue, cl_context context, cl_device_id device);

    int getProgram(cl_program *program, cl_context context, const char *kernelSourceCode);
    int getProgramFromFile(cl_program *program, cl_context context, const char *sourceFilename);
    int getKernel(cl_kernel *kernel, cl_program program, const char *kernelName);
    int getEventTimer(cl_event event,
                      cl_ulong *queuedTime, cl_ulong *submitTime, 
                      cl_ulong *startTime,  cl_ulong *endTime);

private:
    cl_int  _ciErr;
};

#endif // BASICCL_H
