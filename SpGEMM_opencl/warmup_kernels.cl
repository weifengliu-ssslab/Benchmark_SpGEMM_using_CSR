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
