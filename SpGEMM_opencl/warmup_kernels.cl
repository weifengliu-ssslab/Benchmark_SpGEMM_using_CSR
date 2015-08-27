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

inline
void generate_input_simple(__local  uint  *s_key)
{
    int local_id = get_local_id(0);
    int offset_local_id_descending = 2 * get_local_size(0) - 1 - local_id;
    
    // generate two non-duplicated input sequences, A and B, 
    // and copy them into local mem
    s_key[local_id] = local_id * 2 + 1;
    s_key[offset_local_id_descending] = local_id * 2;
}

inline
void coex(__local  uint     *keyA,
          __local  uint     *keyB,
          const int dir)
{
    uint t;
    
    if ((*keyA > *keyB) == dir)
    {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
    }
}

inline
void bitonic_simple(__local uint  *s_key,
                            int            size)
{
    int local_id = get_local_id(0);
    int pos;

    for (uint stride = size >> 1; stride > 0; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        pos = 2 * local_id - (local_id & (stride - 1));
        coex(&s_key[pos], &s_key[pos + stride], 1);
    }
}

__kernel
void warmup(__local  uint  *s_key)
{
    // PHASE 1. DATA GENERATING
    generate_input_simple(s_key);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // PHASE 2. MERGING
    // bitonic sort
    bitonic_simple(s_key, get_local_size(0) * 2);
}
