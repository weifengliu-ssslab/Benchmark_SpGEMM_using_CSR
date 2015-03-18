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

#include "mmio.h"
#include "common.h"
#include "bhsparse.h"

int benchmark_spgemm(char *dataset_name, bool *platforms, bool use_host_mem)
{
    int err = 0;

    cout << "dataset_name = " << dataset_name << endl;

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, k, nnzA;
    index_type *csrRowPtrA;
    index_type *csrColIdxA;
    value_type *csrValA;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(dataset_name, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner." << endl;
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Data type is 'COMPLEX', only read its $real$ part. " << endl;
        isComplex = 1;
        //return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &k, &nnzA_mtx_report)) != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //cout << "symmetric = true" << endl;
    }
    else
    {
        //cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    value_type *csrValA_tmp    = (value_type *)malloc(nnzA_mtx_report * sizeof(value_type));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fxval;
        int ival;

        if (isReal)
        {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fxval);
        }
        else if (isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (value_type *)malloc(nnzA * sizeof(value_type));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);







    int n = k;

    // A
    for (int i = 0; i < nnzA; i++)
    {
        //srand(time(NULL));
        csrValA[i] = 1.0; //( rand() % 9 ) + 1;
    }

    // B = A
    int nnzB = nnzA;
    index_type *csrColIdxB = (index_type *)malloc(nnzB * sizeof(index_type));
    memcpy(csrColIdxB, csrColIdxA, nnzB * sizeof(index_type));
    index_type *csrRowPtrB = (index_type *)malloc((m+1) * sizeof(index_type));
    memcpy(csrRowPtrB, csrRowPtrA, (m+1) * sizeof(index_type));
    value_type *csrValB    = (value_type *)malloc(nnzB * sizeof(value_type));
    memcpy(csrValB, csrValA, nnzB * sizeof(value_type));

    cout << " ( n = " << m << ", nnz = " << nnzA << " ) " << endl;

    // C
    index_type *csrRowPtrC = (index_type *)  malloc((m+1) * sizeof(index_type));

    // start spgemm
    bhsparse *bh_sparse = new bhsparse();

    err = bh_sparse->initPlatform(platforms);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->initData(m, k, n,
                          nnzA, csrValA, csrRowPtrA, csrColIdxA,
                          nnzB, csrValB, csrRowPtrB, csrColIdxB,
                          csrRowPtrC, use_host_mem);
    if(err != BHSPARSE_SUCCESS) return err;

    for (int i = 0; i < 3; i++)
    {
        err = bh_sparse->warmup();
        if(err != BHSPARSE_SUCCESS) return err;
    }

    err = bh_sparse->spgemm();
    if(err != BHSPARSE_SUCCESS) return err;

    // read back C
    int nnzC = bh_sparse->get_nnzC();
    index_type *csrColIndC = (index_type *)  malloc(nnzC  * sizeof(index_type));
    value_type *csrValC    = (value_type *)  malloc(nnzC  * sizeof(value_type));

    err = bh_sparse->get_C(csrColIndC, csrValC);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->free_mem();
    if(err != BHSPARSE_SUCCESS) return err;
    err = bh_sparse->freePlatform();
    if(err != BHSPARSE_SUCCESS) return err;

    // evaluate C
    cout << "nnzC = " << nnzC << endl;

    // free A, B and C
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);
    free(csrColIdxB);
    free(csrValB);
    free(csrRowPtrB);
    free(csrColIndC);
    free(csrValC);
    free(csrRowPtrC);

    return BHSPARSE_SUCCESS;
}

int test_small_spgemm(bool *platforms, bool use_host_mem)
{
    int err = 0;

    int m = 4;
    int k = 6;
    int n = 4;
    int nnzA = 6;
    int nnzB = 7;

    index_type A_row_offsets[4+1];
    index_type A_column_indices[6];
    value_type A_values[6];
    index_type B_row_offsets[6+1];
    index_type B_column_indices[7];
    value_type B_values[7];

    A_row_offsets[0] = 0;
    A_row_offsets[1] = 1;
    A_row_offsets[2] = 4;
    A_row_offsets[3] = 5;
    A_row_offsets[4] = 6;

    A_column_indices[0] = 0;
    A_column_indices[1] = 1;
    A_column_indices[2] = 2;
    A_column_indices[3] = 3;
    A_column_indices[4] = 3;
    A_column_indices[5] = 1;

    for (int i = 0; i < nnzA; i++)
        A_values[i] = (value_type)((i + 1) * 10);

    index_type *csrColIndA = &A_column_indices[0];
    index_type *csrRowPtrA = &A_row_offsets[0];
    value_type *csrValA    = &A_values[0];

    // B

    B_row_offsets[0] = 0;
    B_row_offsets[1] = 1;
    B_row_offsets[2] = 3;
    B_row_offsets[3] = 5;
    B_row_offsets[4] = 5;
    B_row_offsets[5] = 5;
    B_row_offsets[6] = 7;

    B_column_indices[0] = 0;
    B_column_indices[1] = 1;
    B_column_indices[2] = 3;
    B_column_indices[3] = 0;
    B_column_indices[4] = 1;
    B_column_indices[5] = 1;
    B_column_indices[6] = 3;

    for (int i = 0; i < nnzB; i++)
        B_values[i] = (value_type)(i + 1);

    index_type *csrColIndB = &B_column_indices[0];
    index_type *csrRowPtrB = &B_row_offsets[0];
    value_type *csrValB    = &B_values[0];

    // C
    index_type *csrRowPtrC = (index_type *)  malloc((m+1) * sizeof(index_type));

    bhsparse *bh_sparse = new bhsparse();

    err = bh_sparse->initPlatform(platforms);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->initData(m, k, n,
                    nnzA, csrValA, csrRowPtrA, csrColIndA,
                    nnzB, csrValB, csrRowPtrB, csrColIndB,
                    csrRowPtrC, use_host_mem);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->spgemm();
    if(err != BHSPARSE_SUCCESS) return err;

    // C
    int nnzC = bh_sparse->get_nnzC();
    index_type *csrColIndC = (index_type *)  malloc(nnzC  * sizeof(index_type));
    value_type *csrValC    = (value_type *)  malloc(nnzC  * sizeof(value_type));

    err = bh_sparse->get_C(csrColIndC, csrValC);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->free_mem();
    if(err != BHSPARSE_SUCCESS) return err;
    err = bh_sparse->freePlatform();
    if(err != BHSPARSE_SUCCESS) return err;

    // evaluate C
    cout << "nnzC = " << nnzC << endl;

    free(csrColIndC);
    free(csrRowPtrC);
    free(csrValC);

    return BHSPARSE_SUCCESS;
}

int main(int argc, char ** argv)
{
    // get dataset

    // read arguments
    bool *platforms = (bool *)malloc(sizeof(bool) * NUM_PLATFORMS);
    memset(platforms, 0, sizeof(bool) * NUM_PLATFORMS);
    int argi = 1;
    int task_id = 0;
    char *dataset_name;
    bool use_host_mem = false;

    // read method
    if(argc > argi)
    {
        char* option = argv[argi];
        argi++;
        if (strcmp(option, "-cuda") == 0)
            platforms[BHSPARSE_CUDA] = true;
        else if (strcmp(option, "-opencl") == 0)
            platforms[BHSPARSE_OPENCL] = true;
        else if (strcmp(option, "-opencl-hcmp") == 0)
        {
            platforms[BHSPARSE_OPENCL] = true;
            use_host_mem = true;
        }
    }

    // read task
    if (argc > argi)
    {
        char* option = argv[argi];
        argi++;
        if (strcmp(option, "-spgemm") == 0)
        {
            task_id = 0;
            if(argc > argi)
            {
                dataset_name = argv[argi];
                argi++;
            }
        }
    }

    // launch testing dataset or benchmark datasets
    cout << "------------------------" << endl;
    int err = 0;
    if (task_id == 0)
    {
        if (strcmp(dataset_name, "0") == 0)
            err = test_small_spgemm(platforms, use_host_mem);
        else
            err = benchmark_spgemm(dataset_name, platforms, use_host_mem);
    }

    if (err != BHSPARSE_SUCCESS) cout << "Found an err, code = " << err << endl;
    cout << "------------------------" << endl;

    free(platforms);

    return 0;
}
