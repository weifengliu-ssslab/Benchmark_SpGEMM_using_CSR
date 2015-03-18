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

#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>

#include "common.h"
#include "bhsparse.h"
#include "ref.h"

typedef cusp::csr_matrix<index_type,value_type,cusp::host_memory>   CSRHost;

int benchmark_spgemm(char *dataset_name,
                   bool *platforms)
{
    CSRHost A;
    CSRHost B;

    if (strcmp(dataset_name, "1") == 0)
    {
        cusp::gallery::poisson5pt(A, 256, 256);
        cusp::gallery::poisson5pt(B, 256, 256);
        cout << "2D FD, 5-point. ";
    }
    else if (strcmp(dataset_name, "2") == 0)
    {
        cusp::gallery::poisson9pt(A, 256, 256);
        cusp::gallery::poisson9pt(B, 256, 256);
        cout << "2D FE, 9-point. ";
    }
    else if (strcmp(dataset_name, "3") == 0)
    {
        cusp::gallery::poisson7pt(A, 51, 51, 51);
        cusp::gallery::poisson7pt(B, 51, 51, 51);
        cout << "3D FD, 7-point. ";
    }
    else if (strcmp(dataset_name, "4") == 0)
    {
        cusp::gallery::poisson27pt(A, 51, 51, 51);
        cusp::gallery::poisson27pt(B, 51, 51, 51);
        cout << "3D FE, 27-point. ";
    }
    else
    {
        cout << dataset_name;
        cusp::io::read_matrix_market_file(A, dataset_name);

        ref *ref_comp = new ref();
        ref_comp->csr_sort_indices<index_type, value_type>(A.num_rows, &A.row_offsets[0], &A.column_indices[0], &A.values[0]);

        B = A;
    }

    int m = A.num_rows;
    int k = A.num_cols;
    int n = B.num_cols;

    // A
    int nnzA = A.num_entries;
    index_type *csrColIndA = &A.column_indices[0];
    index_type *csrRowPtrA = &A.row_offsets[0];
    value_type *csrValA    = &A.values[0];

    srand(time(NULL));
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = ( rand() % 9 ) + 1;
    }

    // B
    int nnzB = B.num_entries;
    index_type *csrColIndB = &B.column_indices[0];
    index_type *csrRowPtrB = &B.row_offsets[0];
    value_type *csrValB    = &B.values[0];

    for (int i = 0; i < nnzB; i++)
    {
        csrValB[i] = ( rand() % 9 ) + 1;
    }

    cout << " ( n = " << m << ", nnz = " << nnzA << " ) " << endl;

    // C
    index_type *csrRowPtrC = (index_type *)  malloc((m+1) * sizeof(index_type));

    // start spgemm
    int err = 0;
    bhsparse *bh_sparse = new bhsparse();

    err = bh_sparse->initPlatform(platforms);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->initData(m, k, n,
                          nnzA, csrValA, csrRowPtrA, csrColIndA,
                          nnzB, csrValB, csrRowPtrB, csrColIndB,
                          csrRowPtrC);
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
    ref *ref_comp = new ref();
    ref_comp->compData(A, B, m, nnzC, csrRowPtrC, csrColIndC, csrValC);

    // free C
    free(csrColIndC);
    free(csrValC);
    free(csrRowPtrC);

    return BHSPARSE_SUCCESS;
}

int test_small_spgemm(bool *platforms)
{
    int err = 0;

    int m = 4;
    int k = 6;
    int n = 4;
    int nnzA = 6;
    int nnzB = 7;

    CSRHost A(m, k, nnzA);

    A.row_offsets[0] = 0;
    A.row_offsets[1] = 1;
    A.row_offsets[2] = 4;
    A.row_offsets[3] = 5;
    A.row_offsets[4] = 6;

    A.column_indices[0] = 0;
    A.column_indices[1] = 1;
    A.column_indices[2] = 2;
    A.column_indices[3] = 3;
    A.column_indices[4] = 3;
    A.column_indices[5] = 1;

    for (int i = 0; i < nnzA; i++)
        A.values[i] = (value_type)((i + 1) * 10);

    index_type *csrColIndA = &A.column_indices[0];
    index_type *csrRowPtrA = &A.row_offsets[0];
    value_type *csrValA    = &A.values[0];

    // B
    CSRHost B(k, n, nnzB);

    B.row_offsets[0] = 0;
    B.row_offsets[1] = 1;
    B.row_offsets[2] = 3;
    B.row_offsets[3] = 5;
    B.row_offsets[4] = 5;
    B.row_offsets[5] = 5;
    B.row_offsets[6] = 7;

    B.column_indices[0] = 0;
    B.column_indices[1] = 1;
    B.column_indices[2] = 3;
    B.column_indices[3] = 0;
    B.column_indices[4] = 1;
    B.column_indices[5] = 1;
    B.column_indices[6] = 3;

    for (int i = 0; i < nnzB; i++)
        B.values[i] = (value_type)(i + 1);

    index_type *csrColIndB = &B.column_indices[0];
    index_type *csrRowPtrB = &B.row_offsets[0];
    value_type *csrValB    = &B.values[0];

    // C
    index_type *csrRowPtrC = (index_type *)  malloc((m+1) * sizeof(index_type));

    bhsparse *bh_sparse = new bhsparse();

    err = bh_sparse->initPlatform(platforms);
    if(err != BHSPARSE_SUCCESS) return err;

    err = bh_sparse->initData(m, k, n,
                    nnzA, csrValA, csrRowPtrA, csrColIndA,
                    nnzB, csrValB, csrRowPtrB, csrColIndB,
                    csrRowPtrC);
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
    ref *ref_comp = new ref();
    ref_comp->compData(A, B, m, nnzC, csrRowPtrC, csrColIndC, csrValC);

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

    // read method
    if(argc > argi)
    {
        char* option = argv[argi];
        argi++;
        if (strcmp(option, "-cuda") == 0)
            platforms[BHSPARSE_CUDA] = true;
        else if (strcmp(option, "-opencl") == 0)
            platforms[BHSPARSE_OPENCL] = true;
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
            err = test_small_spgemm(platforms);
        else
            err = benchmark_spgemm(dataset_name, platforms);
    }

    if (err != BHSPARSE_SUCCESS) cout << "Found an err, code = " << err << endl;
    cout << "------------------------" << endl;

    free(platforms);

    return 0;
}
