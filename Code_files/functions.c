# include "header.h"

// ----------------------------------------------------------------
// Convert a temporal datum in seconds to (hours, minutes, seconds)
// ----------------------------------------------------------------
// temps Žs la dada temporal en segons
// res Žs el vector amb la dada en el format desitjat

void convert_time( double time, int res[3] ) {
    
    int time_int, hours, minutes, seconds;
    
    time_int = (int) time;
    
    // Minutes
    minutes = time_int/60;
    
    // Division residue
    seconds = time_int % 60;
    
    // Hours
    hours = minutes/60;
    
    // Division residue
    minutes = minutes % 60;
    
    // Result
    res[0] = hours;
    res[1] = minutes;
    res[2] = seconds;
    
    return;
}

// ----------------------
// Allocate an int vector
// ----------------------
int * allocate_vector_int ( int n, char *name ) {
    
    int *v;
    
    v = (int *) calloc( n, sizeof(int));
        
    if ( v == NULL ) {
        printf("No memory for vector %s\n", name );
        exit(1);
    }

    return ( v );
}

// ----------------------
// Allocate an int matrix
// ----------------------
int ** allocate_matrix_int ( int n, int m, char *name ) {
    
    int **mat, i;
    
    mat = (int **) malloc( n * sizeof(int *));
    
    if ( mat == NULL ) {
        printf("No memory for matrix %s\n", name );
        exit(1);
    }
    
    for ( i = 0; i < n; i++ ) {
        
        mat[i] = (int *) calloc( m, sizeof(int));
        
        if ( mat[i] == NULL ) {
            printf("No memory for matrix %s\n", name );
            exit(1);
        }
    }
    
    return ( mat );
}

// ------------------------
// Allocate a double vector
// ------------------------
double * allocate_vector_double ( int n, char *name ) {
    
    double *v;
    
    v = (double *) calloc( n, sizeof(double));
    
    if ( v == NULL ) {
        printf("No memory for vector %s\n", name );
        exit(1);
    }
    
    return ( v );
}

// ------------------------
// Allocate a double matrix
// ------------------------
double ** allocate_matrix_double ( int n, int m, char *name ) {
    
    int i;
    double **mat;
    
    mat = (double **) malloc( n * sizeof(double *));
    
    if ( mat == NULL ) {
        printf("No memory for matrix %s\n", name );
        exit(1);
    }
    
    for ( i = 0; i < n; i++ ) {
        
        mat[i] = (double *) calloc( m, sizeof(double));
        
        if ( mat[i] == NULL ) {
            printf("No memory for matrix %s\n", name );
            exit(1);
        }
    }
    
    return ( mat );
}

// --------------------
// Copy a double vector
// --------------------
void copy_vector_double( int N, double *source, double *dest ) {
 
    int i;
    
    for ( i = 0; i < N; i++ )
        dest[i] = source[i];
    
    return;
}

// -----------
// Copy a file
// -----------
void copy_file ( char *source_file, char *target_file ) {
    
    char ch;
    FILE *source, *target;
    
    source = fopen(source_file, "r");
    
    if ( source == NULL ) {
        printf("Source file not found:\n%s\n\n", source_file );
        exit(EXIT_FAILURE);
    }
    
    target = fopen(target_file, "w");
    
    if( target == NULL ) {
        fclose(source);
        printf("Target file not found:\n%s\n\n", target_file );
        exit(EXIT_FAILURE);
    }
    
    while( ( ch = fgetc(source) ) != EOF )
        fputc(ch, target);
    
    //printf("File copied successfully.\n");
    
    fclose(source);
    fclose(target);
    
    return;
}

// -----------
// Open a file
// -----------
// mode = "w" (write), "r" (read), "a" (append), ...
FILE * open_file( char *directory, char *name, char *mode ) {
    
    FILE *f;
    char complete_name[300];
    
    sprintf( complete_name, "%s/%s", directory, name );
    
    f = fopen( complete_name, mode );
    
    if ( f == NULL ) {
        printf("\nNo memory for file\n%s\n\n", complete_name );
        exit(1);
    }
    
    return (f);
}

// ------------------------
// Print a vector in a file
// ------------------------
void print_vector( FILE *f, double *v, int n ) {
    
    int i;
    
    for ( i = 0; i < n; i++ ) {
        fprintf( f, "%8.3f\n", v[i] );
    }
    return;
}

// ----------------------------
// Print a gsl_matrix in a file
// ----------------------------
void print_matrix_gsl( FILE *f, gsl_matrix *Mat ) {
    
    int i, j;
    
    int n = (int) Mat->size1;
    int m = (int) Mat->size2;
    
    for ( i = 0; i < n; i++ ) {
        for ( j = 0; j < m; j++ )
            fprintf( f, "%8.3f", gsl_matrix_get( Mat, i, j ) );
        fprintf( f, "\n" );
    }
    return;
}

// ----------------------------------------------
// Print a gsl_matrix to read it with Mathematica
// ----------------------------------------------
void print_matrix_gsl_mathematica( FILE *f, gsl_matrix *Mat ) {
    
    int n, m, i, j;
    
    n = (int) Mat->size1;
    m = (int) Mat->size2;
    
    fprintf( f, "{\n" );
    
    for ( i = 0; i < n; i++ ) {
        
        fprintf( f, "{" );
        for ( j = 0; j < m; j++ ) {
         
            if ( j < (m-1) )
                fprintf( f, "%.8f, ", gsl_matrix_get( Mat, i, j ) );
            
            else
                fprintf( f, "%.8f", gsl_matrix_get( Mat, i, j ) );
        }
        
        if ( i < (m-1) )
            fprintf( f, "},\n" );
        else
            fprintf( f, "}\n" );
    }
    
    fprintf( f, "}" );
    return;
}

// ------------------------------
// Print a gsl_vector in terminal
// ------------------------------
void print_vector_gsl_terminal( gsl_vector *v ) {
    
    int i, n;
    
    n = (int) v->size;
    
    for ( i = 0; i < n; i++ ) {
        printf( "%12.2e", gsl_vector_get( v, i ) );
    }
    printf( "\n" );
    
    return;
}

// ------------------------------
// Print a gsl_matrix in terminal
// ------------------------------
void print_matrix_gsl_terminal( gsl_matrix *Mat ) {
    
    int i, j, n, m;
    
    n = (int) Mat->size1;
    m = (int) Mat->size2;
    
    for ( i = 0; i < n; i++ ) {
        for ( j = 0; j < m; j++ )
            printf( "%12.3e", gsl_matrix_get( Mat, i, j ) );
        printf( "\n" );
    }
    printf( "\n" );
    
    return;
}

// ----------------------------------
// Print a gsl_matrix_int in terminal
// ----------------------------------
void print_matrix_gsl_int_terminal( gsl_matrix_int *Mat ) {
    
    int i, j, n, m;
    
    n = (int) Mat->size1;
    m = (int) Mat->size2;
    
    for ( i = 0; i < n; i++ ) {
        for ( j = 0; j < m; j++ )
            printf( "%d ", gsl_matrix_int_get( Mat, i, j ) );
        printf( "\n" );
    }
    printf( "\n" );
    
    return;
}

// ---------------------------------
// Print a double** matrix in a file
// ---------------------------------
void print_matrix_double( FILE *f, double **Mat, int n, int m ) {
    
    int i, j;
    
    for ( i = 0; i < n; i++ ) {
        for ( j = 0; j < m; j++ )
            fprintf( f, "%8.3f", Mat[i][j] );
        fprintf( f, "\n" );
    }
    return;
}

// --------------------------------
// Rescale the original dMat matrix
// --------------------------------
// The non-existing connections are given a weight d0
// The other connections are multiplied by dd
void rescale_dMat ( double dd, double d0, gsl_matrix *dMat, gsl_matrix *dMat_or ) {
    
    int N, i, j;
    double dm;
    
    N = (int) dMat_or->size1;
    
    if ( dMat->size1 != N || dMat->size2 != N || dMat_or->size2 != N ) {
        printf("There is a problem with dimensions:\ndMat_or is %d x %d and dMat is %d x %d.\n\n", (int)dMat_or->size1, (int)dMat_or->size2, (int)dMat->size1, (int)dMat->size2 );
        exit(1);
    }
    
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < N; j++ ) {
            
            dm = gsl_matrix_get( dMat_or, i, j );
            if ( dm > 1e-14 )
                gsl_matrix_set( dMat, i, j, dd * dm );
            else
                gsl_matrix_set( dMat, i, j, d0 );
        }
    }
    return;
}

// ------------------------------
// Perturb a given node partition
// ------------------------------

// For f in [0,1], we do the following f*N times:
//      1. choose 2 nodes at random
//      2. switch their memberships
// The result is stored in the vector "groups", which gives the index of the group to which each node belongs

void perturb_node_partition ( int N, int *groups, gsl_vector_int *group_indices, double f ) {
 
    int n, i, j, g, flip, index, nflips;
    
    n = (int) group_indices->size;
    
    if ( n == 1 ) {
        printf("I cannot perturb the partition because there is only one group!\n\n" );
        return;
    }
    
    // Original groups
    g = 0;
    index = gsl_vector_int_get( group_indices, 1 );
    
    for ( i = 0; i < N; i++ ) {
        
        if ( i >= index ) {
            g++;
            
            if ( g < (n-1) )
                index = gsl_vector_int_get( group_indices, g+1 );
            else
                index = N;
        }
        groups[i] = g;
    }
    
    // Iterations
    nflips = (int) ( f*N );
    
    for ( flip = 0; flip < nflips; flip++ ) {
    
        // Choose 2 nodes at random
        i = (int) RANDOM_INT(N);
        j = (int) RANDOM_INT(N);
        
        g = groups[i];
        groups[i] = groups[j];
        groups[j] = g;
    }
    
    return;
}

// ---------------------------------------------
// Reorder dMat according to the groups provided
// ---------------------------------------------

// groups: index of the group to which each node belongs
// map: node permutation corresponding to the perturbation:
//      map[i]: original index of the node that finally occupies position i

void reorder_dMat ( gsl_matrix *dMat, int *groups, int *map ) {
    
    int N, i, j, i2, j2;
    double aux;
    double *groups_double, *mapping;
    gsl_matrix *dMat_copy;
    
    N = (int) dMat->size1;
    
    mapping = allocate_vector_double( N, "mapping" ); // double version of map
    groups_double = allocate_vector_double( N, "groups_double" );
    
    for ( i = 0; i < N; i++ ) {
        groups_double[i] = groups[i];
        mapping[i] = i;
    }
    
    gsl_sort2( groups_double, 1, mapping, 1, N );
    
    // Matrix copy
    dMat_copy = gsl_matrix_calloc( N, N );
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < N; j++ )
            gsl_matrix_set( dMat_copy, i, j, gsl_matrix_get( dMat, i, j ) );
    }
    
    // Matrix reordering
    for ( i = 0; i < N; i++ ) {
        
        i2 = (int) mapping[i];
        
        for ( j = 0; j < N; j++ ) {
            
            j2 = (int) mapping[j];
            
            aux = gsl_matrix_get ( dMat_copy, i2, j2 );
            gsl_matrix_set( dMat, i, j, aux );
        }
    }
    
    // Copy mapping in map
    for ( i = 0; i < N; i++ )
        map[i] = (int) mapping[i];
    
    free( mapping );
    free( groups_double );
    gsl_matrix_free( dMat_copy );
    
    return;
}

// ---------------------------------------------
// Print the vectors that define the observables
// ---------------------------------------------

void print_vectors_observables( struct params_all *params, char *name, char *directory ) {
    
    int N, nu, i, n, index, *ns;
    gsl_vector_int *group_indices;
    gsl_matrix *mMat;
    FILE *fout;
    
    ns = params->ns;
    group_indices = params->group_indices;
    mMat = params->mMat;
    
    n = (int) mMat->size1;
    N = (int) mMat->size2;
    
    fout = open_file( directory, name, "w" );
    
    fprintf( fout, "# network size: %d\n# number of groups: %d\n", N, n );
    fprintf( fout, "# group sizes:\n " );
    for ( i = 0; i < n; i++ )
        fprintf( fout, "%d ", ns[i] );
    
    fprintf( fout, "\n\n# observable vectors (by rows):\n" );
    
    for( nu = 0; nu < n; nu++ ) {
        
        index = gsl_vector_int_get( group_indices, nu );
        
        for ( i = 0; i < ns[nu]; i++ )
            fprintf( fout, "%10.10f ", gsl_matrix_get( mMat, nu, i + index ) );
        fprintf( fout, "\n" );
    }
    
    fclose( fout );
    
    return;
}

// --------------------------------------------------------------
// Read and create the original dMat matrix, given as an edgelist
// --------------------------------------------------------------
void read_and_create_dMat_or ( int *N, gsl_matrix **dMat_or, char *name, char *directory ) {
    
    int i, j, N2;
    double w;
    char chain[100], cc;
    FILE *f;
    
    f = open_file( directory, name, "r" );
    
    // Read size N
    fgets( chain, 300, f );
    fscanf( f, "%c", &cc );
    fscanf( f, "%d", &N2 );
    *N = N2;
    
    // Allocate dMat_or
    *dMat_or = gsl_matrix_calloc( *N, *N );
    
    fgets( chain, 300, f );
    fgets( chain, 300, f );
    
    do {
        fscanf( f, "%d", &j );
        fscanf( f, "%d", &i );
        fscanf( f, "%lf", &w );
        gsl_matrix_set( *dMat_or, i, j, w );
    }
    while ( feof(f) == 0 ); // warning: for this to work there must NOT be any space at the end of the file
    
    fclose(f);
    return;
}

// -----------------------------------------
// Read the number and sizes of communitites
// -----------------------------------------
void read_communities ( int *n, int **sizes, gsl_vector_int **group_indices, char *name, char *directory ) {
    
    int i, m, acum = 0;
    char chain[100], cc;
    FILE *f;
    
    f = open_file( directory, name, "r" );
    
    fgets( chain, 300, f );
    fscanf( f, "%c", &cc );
    fscanf( f, "%d", n );
    
    fgets( chain, 300, f );
    fgets( chain, 300, f );
    fgets( chain, 300, f );
    
    *sizes = allocate_vector_int( *n, "*sizes" );
    *group_indices = gsl_vector_int_alloc( *n ); // micro index where each group starts
    
    for ( i = 0; i < *n; i++ ) {
        
        fscanf( f, "%d", &m );
    
        (*sizes)[ i ] = m;
        gsl_vector_int_set( *group_indices, i, acum );
        acum += m;
    }

    fclose(f);
    return;
}

// ---------------------------------------------------
// Compute the parameters of the homogeneous reduction
// ---------------------------------------------------
void compute_parameters_homogeneous ( struct params_all *params ) {
    
    int i, j, nu, rho, n_nu, n_rho, index_nu, index_rho;
    double mean_k;
    
    int n = params->n;
    int *ns = params->ns;
    
    gsl_matrix *dMat = params->dMat;
    gsl_matrix *mMat = params->mMat;
    gsl_matrix *deltaMat = params->deltaMat;
    gsl_matrix *muMat = params->muMat;
    gsl_matrix *lambdaMat = params->lambdaMat;
    gsl_vector_int *group_indices = params->group_indices;
    
    // Initialize matrices
    gsl_matrix_set_zero( mMat );
    
    // Average connectivity between groups
    for ( nu = 0; nu < n; nu++ ) {
    
        n_nu = ns[nu];
        index_nu = gsl_vector_int_get( group_indices, nu );
        
        for ( rho = 0; rho < n; rho++ ) {
        
            n_rho = ns[rho];
            index_rho = gsl_vector_int_get( group_indices, rho );
            
            mean_k = 0;
            for ( i = 0; i < n_nu; i++ ) {
                
                for ( j = 0; j < n_rho; j++ )
                    mean_k += gsl_matrix_get( dMat, i + index_nu, j + index_rho );
            }
            mean_k /= n_nu;
            
            // Introduce the data of the reduced system
            gsl_matrix_set( deltaMat, nu, rho, mean_k );
            gsl_matrix_set( muMat, nu, rho, mean_k );
            gsl_matrix_set( lambdaMat, nu, rho, mean_k );
        }
        
        for ( i = 0; i < n_nu; i++ )
            gsl_matrix_set( mMat, nu, i + index_nu, 1./n_nu );
    }
    
    return;
}

// -----------------------------------------------------------------
// Compute the parameters of the degree-based reduction (Gao et al.)
// -----------------------------------------------------------------
// Only valid when n=1
// The observable vector is (kout_1, ..., kout_N) / <k>, where <k> = Sum_i kout_i
// The parameter of the reduced system is beta = <kout kin> / <k>, where <kout kin> = Sum_i ( kout_i * kin_i )

void compute_parameters_degree ( struct params_all *params ) {
    
    int i, j;
    double *kIn, *kOut, mean_k, kIn_kOut;
    
    int N = params->N;
    int n = params->n;
    gsl_matrix *dMat = params->dMat;
    gsl_matrix *mMat = params->mMat;
    gsl_matrix *deltaMat = params->deltaMat;
    gsl_matrix *muMat = params->muMat;
    gsl_matrix *lambdaMat = params->lambdaMat;
    
    if ( n != 1 ) {
        printf( "Error: to compute the parameters of the degree method, n must be 1.\n\n" );
        exit(1);
    }
    
    // In- and out-degrees of nodes
    kIn = allocate_vector_double( N, "kIn" );
    kOut = allocate_vector_double( N, "kOut" );
    
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < N; j++ ) {
            kIn[i] += gsl_matrix_get( dMat, i, j );
            kOut[i] += gsl_matrix_get( dMat, j, i );
        }
    }
    
    // Compute <k> and <kIn kOut>
    mean_k = 0;
    kIn_kOut = 0;
    
    for ( i = 0; i < N; i++ ) {
        
        mean_k += kIn[i];
        kIn_kOut += kIn[i] * kOut[i];
    }
    
    // Introduce the data of the reduced system
    gsl_matrix_set( deltaMat, 0, 0, kIn_kOut / mean_k );
    gsl_matrix_set( muMat, 0, 0, kIn_kOut / mean_k );
    gsl_matrix_set( lambdaMat, 0, 0, kIn_kOut / mean_k );
    
    for ( i = 0; i < N; i++ )
        gsl_matrix_set( mMat, 0, i, kOut[i] / mean_k );
    
    free( kIn );
    free( kOut );
    
    return;
}

// ------------------
// 1-norm of a vector
// ------------------
double vector_norm_1 ( gsl_vector *v ) {
 
    int i, N;
    double norm = 0;
    
    N = (int) v->size;
    
    for ( i = 0; i < N; i++ )
        norm += fabs( gsl_vector_get( v, i ) );

    return (norm);
}

// --------------------------------------
// Euclidean distance between two vectors
// --------------------------------------
double euc_dist_vectors( int N, double *u, double *v ) {
 
    int i;
    double dist;
    gsl_vector *dif;
    
    dif = gsl_vector_alloc( N );
    
    // dif = u - v
    for ( i = 0; i < N; i++ )
        gsl_vector_set( dif, i, u[i]-v[i] );
    
    // Euclidean norm of dif
    dist = gsl_blas_dnrm2( dif );
    
    gsl_vector_free( dif );
    
    return ( dist );
}

// ------------------
// 2-norm of a matrix
// ------------------

// The 2-norm of a real matrix is ||A||_2 = sqrt( | lambda_max( A^T A ) | ),
// where lambda_max is the eigenvalue with largest module

double norm_2_matrix ( gsl_matrix *A ) {
    
    int i, m, n;
    double lambda, norm;
    gsl_vector *vep;
    gsl_matrix *AtA;
    
    m = (int) A->size1;
    n = (int) A->size2;
    
    // If the matrix dimension is m x 1 (i.e., it is a column vector), the norm is the 2-norm of the column vector
    if ( n == 1 ) {
    
        gsl_vector *v = gsl_vector_alloc( m );
        for ( i = 0; i < m; i++ )
            gsl_vector_set( v, i, gsl_matrix_get( A, i, 0 ) );
        
        norm = gsl_blas_dnrm2( v ); // Euclidean norm of v
        gsl_vector_free( v );
    }
    
    else {
        
        AtA = gsl_matrix_alloc( n, n );
        vep = gsl_vector_alloc( n );
        
        // Ata = A^T A
        gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1., A, A, 0., AtA );
        
        // Leading eigenvalue
        dominant_eigenv( AtA, vep, &lambda );
        
        // Norm
        norm = sqrt( fabs( lambda ) );
        
        gsl_vector_free( vep );
        gsl_matrix_free( AtA );
    }
    
    return (norm);
}

// ------------------
// 1-norm of a matrix
// ------------------
// The 1-norm of a real matrix is the maximum of the 1-norm of its column vectors
double norm_1_matrix ( gsl_matrix *A ) {
    
    int i, j, m, n;
    double norm, norm0;
    
    m = (int) A->size1;
    n = (int) A->size2;
    
    norm = 0;
    
    for ( j = 0; j < n; j++ ) {
     
        norm0 = 0;
        
        for ( i = 0; i < m; i++ )
            norm0 += fabs( gsl_matrix_get( A, i, j ) );
        
        if ( norm0 > norm )
            norm = norm0;
    }
    
    return (norm);
}

// ------------------------------------------------------------------------------
// Find the leading eigenvector and eigenvalue of a matrix using the power method
// ------------------------------------------------------------------------------
int dominant_eigenv( gsl_matrix *Mat, gsl_vector *vep, double *lambda ) {
 
    int N, status=0, i, iter = 0, attempts = 0, max_attempts = 5, iter_max = 10000000;
    double norm0, error = 1, tol = 1.e-12;
    gsl_vector *vep2;
    
    // Dimension
    if( Mat->size1 != Mat->size2 ) {
        printf( "Error: the matrix must be a squared matrix to find its eigenvalues and eigenvectors.\n\n" );
        exit(1);
    }
    
    N = (int) Mat->size1;
    
    if ( vep->size != N ) {
        printf( "Error: the vector to store the eigenvector must have dimension %d.\n\n", N );
        exit(1);
    }
    
    // Initial vector
    for ( i = 0; i < N; i++ )
        gsl_vector_set( vep, i, 1 + 0.5 * RANDOM ); // it's important not to start with the 0 vector
    
    // Normalize
    norm0 = vector_norm_1(vep);
    gsl_vector_scale( vep, 1. / norm0 );
    
    vep2 = gsl_vector_calloc( N );
    
    while ( error > tol && iter < iter_max && attempts < max_attempts ) {
        
        // vep2 = Mat vep
        gsl_blas_dgemv( CblasNoTrans, 1.0, Mat, vep, 0.0, vep2 );
        
        norm0 = vector_norm_1(vep2);
        
        // If vep2 = 0, change vector
        if ( norm0 < 1.e-18 ) {
            
            for ( i = 0; i < N; i++ )
                gsl_vector_set( vep, i, 1 + 0.5 * RANDOM ); // it's important not to start with the 0 vector
            
            norm0 = vector_norm_1(vep);
            gsl_vector_scale( vep, 1. / norm0 );
            
            iter = 0;
            attempts++;
            
            continue;
        }
        
        // Normalize
        gsl_vector_scale( vep2, 1. / norm0 );
        
        // vep = vep - vep2
        gsl_vector_sub( vep, vep2 );
        error = vector_norm_1(vep);
        
        // vep = vep2
        gsl_vector_memcpy( vep, vep2 );
        
        iter++;
    }
    
    if ( iter == iter_max ) {
        status = 1;
        printf( "I have reached the maximum number of iteration without converging.\nError: %.5e\n\n", error );
    }
    else {
        
        if ( attempts == max_attempts ) {
            printf( "I have used the maximum number of attempts: all the initial vectors have converged to the 0 vector.\n\n" );
            exit(1);
        }
    }
    
    // vep2 = Mat vep
    gsl_blas_dgemv( CblasNoTrans, 1.0, Mat, vep, 0.0, vep2 );
    
    // Eigenvalue (vap)
    // The Rayleigh quotient is computed: vap = vep^T Mat vep / ( vep^T vep )
    // In our case, vap = vep^T vep2 / ( vep^T vep )
    gsl_blas_ddot( vep, vep2, lambda );
    gsl_blas_ddot( vep, vep, &norm0 );
    *lambda = *lambda / norm0;
    
    gsl_vector_free( vep2 );
    
    return ( status );
}

// -------------------------------------------------------------------
// Compute matrix C' for a fixed number of eigenvals and eigenvects, r
// -------------------------------------------------------------------
// indices_veps: indices of the eigenvects (veps) that we use to construct C' (the veps could not be ordered)

void compute_cMat ( gsl_matrix *cMat, int r, int *indices_veps, gsl_matrix ** Ms, gsl_vector **vepsMs, double *vapsMs, int n ) {
    
    int j, s0, s, t0, t, n_nu;
    double c_st, c_st_aux;
    gsl_vector *v_aux1, *v_aux2;
    
    n_nu = (int) Ms[0]->size1;
    
    v_aux1 = gsl_vector_alloc( n_nu );
    v_aux2 = gsl_vector_alloc( n_nu );
    
    gsl_matrix_set_zero( cMat );
    
    // Submatrix C = (c_st)_{s,t} is symmetric, so we only need to compute c_st for t >= s
    for ( s0 = 0; s0 < r; s0++ ) {
        
        s = indices_veps[s0];
        
        for ( t0 = s0; t0 < r; t0++ ) {
            
            t = indices_veps[t0];
            c_st = 0;
            
            for ( j = 0; j < n; j++ ) {
                
                // ....................................................
                // .  v_aux1 = Ms[j] vepsMs[s] - vapsMs[j] vepsMs[s]  .
                // ....................................................
                
                // v_aux1 = Ms[j] vepsMs[s]
                gsl_blas_dgemv( CblasNoTrans, 1.0, Ms[j], vepsMs[s], 0.0, v_aux1 );
                
                // v_aux1 = v_aux1 - vapsMs[j] vepsMs[s]
                gsl_vector_axpby( - vapsMs[j], vepsMs[s], 1., v_aux1 );
                
                // ....................................................
                // .  v_aux2 = Ms[j] vepsMs[t] - vapsMs[j] vepsMs[t]  .
                // ....................................................
                
                // v_aux2 = Ms[j] vepsMs[t]
                gsl_blas_dgemv( CblasNoTrans, 1.0, Ms[j], vepsMs[t], 0.0, v_aux2 );
                
                // v_aux2 = v_aux2 - vapsMs[j] vepsMs[t]
                gsl_vector_axpby( - vapsMs[j], vepsMs[t], 1., v_aux2 );
                
                // ..............................
                // .  b_j * < v_aux1, v_aux2 >  .
                // ..............................
                
                // Scalar product between v_aux1 and v_aux2 stored in c_st_aux
                gsl_blas_ddot( v_aux1, v_aux2, &c_st_aux );
                c_st += c_st_aux;
            }
            
            gsl_matrix_set ( cMat, s0, t0, c_st );
            gsl_matrix_set ( cMat, t0, s0, c_st );
        }
    }
    
    // Last column and last row of matrix C'
    for ( j = 0; j < r; j++ )
        gsl_matrix_set( cMat, j, r, -1. );
    
    for ( j = 0; j < r; j++ )
        gsl_matrix_set( cMat, r, j, 1. );
    
    gsl_matrix_set( cMat, r, r, 0. );
    
    gsl_vector_free( v_aux1 );
    gsl_vector_free( v_aux2 );
    
    return;
}

// -------------------------------------------
// Gauss elimination method for a m x n matrix
// -------------------------------------------

void elimination_Gauss ( gsl_matrix *a ){
    
    int i, i0, j, k;
    int n, m;
    double term, akj, aij, aii, aki, pivot, tol;
    
    tol = 1.e-15;
    
    m = (int) a->size1;
    n = (int) a->size2;
    
    i = 0;
    for( i0 = 0; i0 < n; i0++ ) { // i0 indicates the pivot column
        
        if ( i >= m-1 )
            break;
    
        // Partial Pivoting
        for( k = i+1; k < m; k++ ) {
            
            aii = gsl_matrix_get( a, i, i0 );
            aki = gsl_matrix_get( a, k, i0 );
            
            // If aii (absolute value) is smaller than any of the terms below it
            if( fabs( aii ) < fabs( aki ) ) {
                
                //Swap the rows k and i
                for( j = 0; j < n; j++ ){
                    
                    aij = gsl_matrix_get( a, i, j );
                    akj = gsl_matrix_get( a, k, j );
                    
                    gsl_matrix_set( a, i, j, akj );
                    gsl_matrix_set( a, k, j, aij );
                }
            }
        }
        
        // Now the elements of column i0 from row i are ordered in absolute value.
        // If the pivot a[i][i0] is nonzero, Gauss elimination is performed and we go to the next row (i+1) and the next column (i0+1).
        // If the pivot is 0, the following element in row i becomes the pivot. The same is done in row i with the new i0.
        pivot = gsl_matrix_get( a, i, i0 );
        
        if ( fabs(pivot) > tol ) {
        
            // Gauss elimination
            for( k = i+1; k < m; k++ ) {
                
                term = gsl_matrix_get( a, k, i0 ) / gsl_matrix_get( a, i, i0 ); // term = a[k][i0] / a[i][i0]
                
                // Left to the pivot (including it) we introduce 0s (this prevents error accumulation)
                for( j = 0; j <= i0; j++ )
                    gsl_matrix_set( a, k, j, 0 );
                    
                // Right to the pivot we introduce the corresponding combination
                for( j = i0+1; j < n; j++ ) {
                    
                    akj = gsl_matrix_get( a, k, j );
                    aij = gsl_matrix_get( a, i, j );
                    gsl_matrix_set( a, k, j, akj - term * aij );
                }
            }
            
            i++;
        }
    }
    
    return;
}

// -----------------------------------------------------------------------------------------
// Given a collection of vectors v1, ..., vn, return a maximal subcollection of l.i. vectors
// -----------------------------------------------------------------------------------------

// The vectors are given as a mxn matrix, where m is the vector length
// The rank r is returned and also a vector with the indices of the vectors in the maximal subset

int maximal_li_subset ( gsl_matrix *a, int *indices ) {
    
    int i, j, n, m, min, r;
    double tol, aij;
    gsl_matrix *a_aux;
    
    tol = 1.e-15;
    
    m = (int) a->size1;
    n = (int) a->size2;
    
    min = n;
    if ( m < n )
        min = m;
    
    // Copy matrix (it is destroyed in the process)
    a_aux = gsl_matrix_alloc( m, n );
    gsl_matrix_memcpy( a_aux, a );
    
    // Gauss elimination of the matrix
    elimination_Gauss ( a_aux );
    
    // The indices of the vectors in the maximal subset are the columns where the first nonzero element of every row is located, for the first min(n,m) rows (the others have 0s)
    r = 0;
    for ( i = 0; i < min; i++ ) {
     
        for ( j = 0; j < n; j++ ) {
            aij = gsl_matrix_get( a_aux, i, j );
            
            if ( fabs(aij) > tol )
                break;
        }
        
        if ( j < n ) {
            indices[r] = j;
            r++;
        }
    }
    
    gsl_matrix_free( a_aux );
    
    return (r);
}

// -------------------------------------------
// Compute the pseudoinverse of a m x n matrix
// -------------------------------------------
//
// If m < n, we first transpose M and at the end we transpose the obtained matrix
// (because (M^T)+ = (M+)^T and the process is designed for m >= n)
//
// Let us assume that m >= n.
// The first step is to use gsl to compute a "(thin) singular value decomposition":
//      M = U S V^T, where
//          U is m x n
//          S is n x n and diagonal
//          V is n x n
//
// Then we compute the pseudoinverse of S, S+, that is obtained by inverting its diagonal nonzero elements.
//
// Finally, the pseudoinverse of M is
//      M+ = V S+ U^T
//
// The rank of the original matrix is returned (relative to the tolerance tol)

int pseudoinverse( gsl_matrix *M, gsl_matrix *pinvM ) {

    int i, j, m, n, rank;
    double s, v, tol = 1.e-15;
    gsl_matrix *U, *V, *Maux;
    gsl_vector *sV, *work;
    
    m = (int) M->size1;
    n = (int) M->size2;
    
    if ( pinvM->size1 != n || pinvM->size2 != m ) {
        printf( "Error: in order to compute the pseudoinverse, matrix pinvM should have dimension %d x %d.\n\n", n, m );
        exit(1);
    }
    
    // Impose m >= n
    if ( m < n ) {
        m = (int) M->size2;
        n = (int) M->size1;
    }
    
    // Matrix that will store U after computing the pseudoinverse (we transpose it at the end if needed)
    // Initially it is matrix M (or its transpose)
    U = gsl_matrix_alloc( m, n );
    
    // Transpose if needed
    if ( M->size1 < M->size2 )
        gsl_matrix_transpose_memcpy( U, M );
    
    // Otherwise, copy the matrix because it will be destroyed in the process
    else
        gsl_matrix_memcpy( U, M );
    
    // Singular value decomposition
    V = gsl_matrix_alloc( n, n );
    sV = gsl_vector_alloc( n ); // diagonal of S
    work = gsl_vector_alloc( n );
    
    gsl_linalg_SV_decomp( U, V, sV, work );
    
    // Rank of M = number of nonzero singular values
    rank = 0;
    for ( i = 0; i < n; i++ ) {
        s = gsl_vector_get( sV, i );
        if ( fabs(s) > tol )
            rank++;
    }
    
    // Diagonal of S+
    for ( i = 0; i < n; i++ ) {
        s = gsl_vector_get( sV, i );
        if ( fabs(s) >= tol )
            s = 1./s;
        else
            s = 0.;
        gsl_vector_set( sV, i, s );
    }

    // M+ = V S+ U^T

    // 1) V -> V S+ (columns of V are multiplied by the elements of sV)
    for ( i = 0; i < n; i++ ) {
        s = gsl_vector_get( sV, i );
        
        for ( j = 0; j < n; j++ ) {
            v = gsl_matrix_get( V, j, i );
            gsl_matrix_set( V, j, i, s*v );
        }
    }
    
    // 2) Maux = V * U^T
    Maux = gsl_matrix_alloc( n, m );
    gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1., V, U, 0., Maux );
    
    // If the matrix was transposed, we transpose the result
    if ( M->size1 < M->size2 )
        gsl_matrix_transpose_memcpy( pinvM, Maux );
    
    // Otherwise, we copy the result
    else
        gsl_matrix_memcpy( pinvM, Maux );

    gsl_matrix_free( U );
    gsl_matrix_free( V );
    gsl_matrix_free( Maux );
    gsl_vector_free( sV );
    gsl_vector_free( work );
    
    return( rank );
}

// -----------------------------
// Solve a linear system A x = b
// -----------------------------

// If the system has more than one solution, the solution with minimal norm is given
// If the system does not have a solution, the vector that minimizes the error (according to the Euclidean norm) is given
// In any case, the result is x = A+ b, where A+ is the pseudoinverse of A

// The function also returns:
//   0 -> if the system has exactly one solution
//   1 -> if the system has many solutions
//   2 -> if the system has no solution
// (according to a tolerance tol)

int solve_linear_system ( gsl_matrix *A, gsl_vector *b, gsl_vector *x ) {

    int rank, ret;
    double norm, tol = 1.e-15;
    gsl_matrix *pinvA;
    gsl_vector *c;
    
    if ( x->size != A->size2 ) {
        printf( "The dimension of x should be %d and it is %d.\n\n", (int)A->size2, (int)x->size );
        exit(1);
    }
    
    if ( b->size != A->size1 ) {
        printf( "The dimension of b should be %d and it is %d.\n\n", (int)A->size1, (int)b->size );
        exit(1);
    }
    
    // Pseudoinverse of A
    pinvA = gsl_matrix_alloc( A->size2, A->size1 );
    rank = pseudoinverse( A, pinvA );

    // x = A+ b
    gsl_blas_dgemv( CblasNoTrans, 1., pinvA, b, 0., x );
    
    // Determine whether the system has one or more solutions
    
    // The system has at least one solution if A A+ b = b
    // It has exactly one solution if, also, rank(A) = n, where n is the column number of A
    
    // c = b
    c = gsl_vector_alloc( b->size );
    gsl_vector_memcpy( c, b );
    
    // c <- A x - c = A x - b
    gsl_blas_dgemv( CblasNoTrans, 1., A, x, -1, c );
    norm = gsl_blas_dnrm2( c );
    
    ret = -1;
    if ( norm < tol ) { // We consider that the system has solutions
        if ( rank == (int)A->size2 )
            ret = 0;
        else
            ret = 1;
    }
    
    gsl_matrix_free( pinvA );
    gsl_vector_free( c );
    
    return( ret );
}

// ------------------------------------------------------------
// Generate the matrix C' that we need to find the vector alpha
// ------------------------------------------------------------

int generate_cMat ( gsl_matrix **cMat, int *indices_veps, gsl_matrix ** Ms, gsl_vector **vepsMs, double *vapsMs, int n ) {
    
    int i, j, r, n_nu;
    
    n_nu = (int) vepsMs[0]->size;
    
    // indices_veps stores the indices of the eigenvectors used to construct cMat
    for ( i = 0; i < n; i++ )
        indices_veps[i] = i;
    
    // Introduce the vectors in an auxiliary matrix
    gsl_matrix *a = gsl_matrix_alloc( n_nu, n );
    for ( i = 0; i < n_nu; i++ ) {
        for ( j = 0; j < n; j++ )
            gsl_matrix_set( a, i, j, gsl_vector_get( vepsMs[j], i ) );
    }
    
    // Select, within our vectors, a maximal subset of l.i. vectors
    // r is the number of l.i. vectors
    r = maximal_li_subset ( a, indices_veps );
    gsl_matrix_free( a );
    
    // C' matrix
    *cMat = gsl_matrix_alloc ( r+1, r+1 );
    compute_cMat ( *cMat, r, indices_veps, Ms, vepsMs, vapsMs, n );
    
    return( r );
}

// --------------------------------------------------------
// Compute the average norm or the maximal norm of matrices
// Ai := Ms[i] - vap[i] Id, i = 1, ..., n
// --------------------------------------------------------
double average_or_maximal_norm( int n, gsl_matrix ** Ms, double * vapsMs ) {
    
    int i, j, n_nu;
    double norm, norm_av, norm_max;
    gsl_matrix *Ai;
    
    n_nu = (int) Ms[0]->size1;
    norm_av = 0;
    norm_max = 0;
    
    // If n_nu = 1, the norm of Ai is 0 for all i (because Ms[i] = vap[i])
    // Otherwise, we have to compute it
    if ( n_nu != 1 ) {
        
        Ai = gsl_matrix_alloc( n_nu, n_nu );
        
        for ( i = 0; i < n; i++ ) {
            
            // Ai = Ms[i] - vap[i] Id
            gsl_matrix_memcpy( Ai, Ms[i] );
            for ( j = 0; j < n_nu; j++ )
                gsl_matrix_set( Ai, j, j, gsl_matrix_get( Ai, j, j ) - vapsMs[i] );
            
            // Norm of Ai
            //norm = norm_2_matrix( Ai );
            norm = norm_1_matrix( Ai );
            
            if ( isnan(norm) ) {
                printf( "Matrix with nan norm:\n" );
                print_matrix_gsl_terminal( Ai );
                exit(1);
            }
            
            norm_av += norm;
            
            if ( norm > norm_max )
                norm_max = norm;
        }
        norm_av /= n;
        
        gsl_matrix_free( Ai );
    }
    
    // We can return the average or the maximal norm /////// !!!!!!
    return ( norm_max );
    //return ( norm_av );
}

// ------------------------------------------------
// Generate a random vector v of length n such that
//    * Sum_i(vi) = 1
//    * (1-a)/(n-1) <= vi <= a for all i
// ------------------------------------------------

void random_vector_sum1 ( gsl_vector *v, double a ) {
    
    int i, n, fi = 0;
    double vi, sum;
    
    n = (int) v->size;
    
    if ( n == 1 )
        gsl_vector_set( v, 0, 1. );
    
    else {
        
        // Unif. distributed vector which sums 1 and is in the 1st quadrant (0 <= vi <= 1 for all i)
        while ( fi != 1 ) {
            
            sum = 0.;
            
            for ( i = 0; i < n; i++ ) {
                
                vi = RANDOM_GAMMA( 1., 1. ); // aux is an Exp(lambda), lambda=1
                gsl_vector_set( v, i, vi );
                sum += vi;
            }
            
            // Normalize
            if ( fabs(sum) > 1.e-10 ) {
                fi = 1;
                for ( i = 0; i < n; i++ )
                    gsl_vector_set( v, i, gsl_vector_get(v, i) / sum );
            }
        }
        
        // Modify the vector just obtained so that (1-a)/(n-1) <= vi <= a for all i
        for ( i = 0; i < n; i++ ) {
            vi = gsl_vector_get( v, i );
            vi = ( ( a*n - 1 ) * vi  + (1-a) ) / (n-1);
            gsl_vector_set( v, i, vi );
        }
    }
    
    return;
}

// ---------------------------------------------------------------
// Given a vector v, compute the error associated to the equations
//      Ms[i] v = vap[i] v, i = 1, ..., n
// The error is E = Sum_{i=1,n} || Ms[i] v - vap[i] v ||^2
// ---------------------------------------------------------------

double compute_error ( int n, gsl_matrix ** Ms, double * vapsMs, gsl_vector *v ) {

    int j, n_nu;
    double norm, error = 0.;
    gsl_vector *v_aux;
    
    n_nu = (int) Ms[0]->size1;
    
    v_aux = gsl_vector_alloc( n_nu );
    
    for ( j = 0; j < n; j++ ) {
        
        // v_aux = Ms[j] v
        gsl_blas_dgemv( CblasNoTrans, 1.0, Ms[j], v, 0.0, v_aux );
        
        // v_aux = v_aux - vap[j] v
        gsl_vector_axpby( - vapsMs[j], v, 1., v_aux );
        
        // Euclidean norm of v_aux
        norm = gsl_blas_dnrm2( v_aux );
        
        error += pow( norm, 2 );
    }

    gsl_vector_free( v_aux );

    return (error);
}

// ----------------------------------------------------------------------------
// Generate "nerror" random vectors that sum 1 and compute the resulting errors
// in terms of the equations Ms[i] v = vap[i] v, i = 1, ..., n
// ----------------------------------------------------------------------------
// The error is defined by E = Sum_{i=1,n} || Ms[i] v - vap[i] v ||^2
// The error relative to a given factor is stored
// The vectors have the property that all their components are in the range [ (1-a)/(n_nu-1), a ]

void analyze_error( int n, gsl_matrix ** Ms, double * vapsMs, int nerror, double * errors_random, double a, double factor ) {
    
    int i, n_nu;
    double error, norm;
    gsl_vector *v;
    
    n_nu = (int) Ms[0]->size1;
    v = gsl_vector_alloc( n_nu );
    
    for ( i = 0; i < nerror; i++ ) {
        
        // Generate a vector that sums 1 and with components smaller than a
        random_vector_sum1 ( v, a );
        
        ////////////
        // 1-norm of the vector
        norm = vector_norm_1( v );
        if ( norm > 1.000000000001 ) {
            printf( "The 1-norm of the random vector is larger than 1: %.5f\n", norm );
            //getchar();
        }
        ////////
        
        // Error
        error = compute_error ( n, Ms, vapsMs, v );
        errors_random[i] = error / factor;
    }
    
    gsl_vector_free( v );
    
    return;
}

// ---------------------------------
// Solve the compatibility equations
// ---------------------------------

// If error_analysis = 1, we print in a file an analysis of the error.
// This means computing:
//      * the error associated to the approximate solution found
//      * the errors resulting from taking random vectors instead of the approx. solution.
//
// To compute the error:
// - For each community nu, the error associated to the observable vector of the nu community (Mnu) is
//   error(nu) = Sum_{i=1^n} Ms[i] * Mnu - vap[i] Mnu,
//   and it coincides with the unknown K of the linear system to be solved: error(nu) = K
//
// - The total error is the sum of the errors of the different communities, weighted by the relative
//   size of the community (n_nu/N):
//   error = Sum_nu [ n_nu * error(nu) ] / N

void solve_compatibility_eqs ( struct params_all *params, int error_analysis ) {
    
    int i, j, r, nu, rho, n_nu, n_rho, index_nu, index_rho, status, nerror, error_relative_to_norm;
    int *indices_veps;
    double *alphaV, dij, *vapsMs, total, aux, ki, mean_error, *errors, **errors_random, a, min, factor, norm_pm;
    gsl_matrix *dMat_nu_rho, *dMat_rho_nu, **Ms, *cMat, *auxMat;
    gsl_vector **vepsMs, *bV, *M_nu, *M_rho;
    
    int N = params->N;
    int n = params->n;
    int *ns = params->ns;
    int correction_red_syst = params->correction_red_syst;
    
    gsl_matrix *dMat_gran = params->dMat;
    gsl_matrix *mMat = params->mMat;
    gsl_matrix *deltaMat = params->deltaMat;
    gsl_matrix *muMat = params->muMat;
    gsl_matrix *lambdaMat = params->lambdaMat;
    gsl_vector_int *group_indices = params->group_indices;
    char * directory = params->directory;
    
    // Errors in the comp. eqs.
    // The errors will be normalized by a factor that depends on the average norm of the corresponding matrices
    error_relative_to_norm = 1;
    
    nerror = 1000; // Number of random vectors for the error analysis
    if ( error_analysis == 1 ) {
        
        // Errors for each group
        errors = allocate_vector_double( n, "errors" );
        mean_error = 0.;
        
        // Errors of the random vectors for each group
        errors_random = allocate_matrix_double( n, nerror, "errors_random" );
    }
    
    printf( "Solving the compatibility equations... \n\n" );
    
    // Set mMat matrix to 0
    gsl_matrix_set_zero( mMat );
    
    for ( nu = 0; nu < n; nu++ ) {
        
        n_nu = ns[nu];
        index_nu = gsl_vector_int_get( group_indices, nu );
        
        //printf( "\n------------------------\n nu = %d\n------------------------\n", nu );
        
        // M_nu: observable vector of group nu, of length n_nu
        M_nu = gsl_vector_alloc( n_nu );
        auxMat = gsl_matrix_alloc( n_nu, n_nu );
        
        // We will solve n eigenvector-eigenvalue equations assocoated to n matrices M1, ..., Mn
        // Store the pointers to these matrices in the vector Ms:
        Ms = (gsl_matrix **) malloc( n * sizeof( gsl_matrix *) );
        if ( Ms == NULL ) {
            printf( "No memory for the collection of Mj matrices.\n" );
            exit(1);
        }
        
        // The pointers to their leading eigenvectors and eigenvalues are stored in "vapsMs" and "vepsMs"
        vepsMs = ( gsl_vector ** ) malloc( n * sizeof( gsl_vector * ) );
        if ( vepsMs == NULL ) {
            printf( "No memory for the collection of eigenvectors to Ms matrices.\n" );
            exit(1);
        }
        
        for ( i = 0; i < n; i++ ) {
            Ms[i] = gsl_matrix_alloc ( n_nu, n_nu );
            vepsMs[i] = gsl_vector_alloc( n_nu );
        }
        vapsMs = allocate_vector_double( n, "vapsMs" );
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    Generate the different Ms matrices and find their leading evals and evects
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        for ( rho = 0; rho < n; rho++ ) {
        
            n_rho = ns[rho];
            index_rho = gsl_vector_int_get( group_indices, rho );
            
            // Define the submatrix d_{nu rho}^T, of dimension n_rho x n_nu
            dMat_nu_rho = gsl_matrix_alloc (n_rho, n_nu);
            
            for ( i = 0; i < n_nu; i++ ) {
                for ( j = 0; j < n_rho; j++ ) {
                    
                    dij = gsl_matrix_get( dMat_gran, i + index_nu, j + index_rho );
                    gsl_matrix_set ( dMat_nu_rho, j, i, dij ); // transpose it
                }
            }
            
            // Define the submatrix of dimension n_nu x n_rho:
            //      d_{rho nu}^T    if rho != nu
            //      Id              if rho = nu
            dMat_rho_nu = gsl_matrix_alloc (n_nu, n_rho);
            
            if ( n_nu != n_rho ) {
                
                for ( i = 0; i < n_rho; i++ ) {
                    for ( j = 0; j < n_nu; j++ ) {
                        
                        dij = gsl_matrix_get( dMat_gran, i + index_rho, j + index_nu );
                        gsl_matrix_set ( dMat_rho_nu, j, i, dij ); // transpose it
                    }
                }
            }
            
            else
                gsl_matrix_set_identity( dMat_rho_nu );
            
            // Ms[rho] = d_{rho nu}^T d_{nu rho}^T
            gsl_blas_dgemm ( CblasNoTrans, CblasNoTrans,
                            1.0, dMat_rho_nu, dMat_nu_rho,
                            0.0, Ms[rho] );
            
            // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            // Find the leading eigenvalue and eigenvector using the power method
            // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            dominant_eigenv( Ms[rho], vepsMs[rho], &vapsMs[rho] );
            
            total = 0;
            for ( i = 0; i < n_nu; i++ )
                total += gsl_vector_get( vepsMs[rho], i );
            
            // Normalize so that the sum is 1
            gsl_vector_scale( vepsMs[rho], 1./total );
            
            // Free memory
            gsl_matrix_free( dMat_nu_rho ); 
            gsl_matrix_free( dMat_rho_nu );
        }
        
        // ======================================================================================================
        //  If n = 1, then M_nu = vepsMs[0], the error of the nu case is 0 and we don't need to solve any system.
        //  If n > 1, we have to solve a system of eqs. to find alpha and construct M_nu.
        // ======================================================================================================
        
        if ( n == 1 )
            status = gsl_vector_memcpy( M_nu, vepsMs[0] );
        
        else {
        
            // * * * * * * * * * * * * * * * * * * * * *
            //    Define the system to find alpha
            // * * * * * * * * * * * * * * * * * * * * *
            
            // alpha is the coefficient vector that we need to construct M_nu from the eigenvectors.
            // alpha has dimension r, where r is the maximal number of l.i. vectors among the n eigenvectors found.
            // To construct the system's matrix C', though, we use the n eigenvectors.
            //
            // Assume that vep_1, ..., vep_r are these l.i. vectors (the others are vep_{r+1}, ..., vep_n).
            //
            // We must solve the system: C alpha = K 1v, Sum_i alpha_i = 1
            // for the unknowns alpha = ( alpha_1, ..., alpha_r ) and K, where
            //
            //      C = ( c_st )_{s,t}, 1 <= s,t <= r
            //      c_st := Sum_{j=1^n} * < M_j vep_s - vap_j vep_s, M_j vep_t - vap_j vep_t >,
            //      1v = (1, ..., 1)^T
            //      <u, v> is the scalar product between u and v
            //
            // This system can be rewritten as the linear system
            //
            //      C' x  = bV,     where
            //
            //         bV = ( 0, ..., 0, 1 )^T
            //
            //         x  = ( alpha_1, ..., alpha_r, K )
            //
            //              ( c_11 ... c_1r -1 )
            //              (  .        .    . )
            //         C' = (  .        .    . )
            //              (  .        .    . )
            //              ( c_r1 ... c_rr -1 )
            //              (  1   ...  1    0 )
            
            // We start by defining the matrix C'
            indices_veps = allocate_vector_int( n, "indices_veps" );
            r = generate_cMat ( &cMat, indices_veps, Ms, vepsMs, vapsMs, n );
            
            alphaV = allocate_vector_double( r, "alphaV" );
            bV = gsl_vector_alloc( r+1 );
            
            // Vector bV = ( 0, ..., 0, 1 )^T
            for ( i = 0; i < r; i++ )
                gsl_vector_set( bV, i, 0. );
            gsl_vector_set( bV, r, 1 );
            
            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            //    Solve the system C' x = bV using the pseudoinverse
            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            
            // This method gives a solution even if C' is singular
            // The solver function returns:
            //   0 -> if the system has exactly one solution
            //   1 -> if the system has many solutions
            //   2 -> if the system has no solution

            gsl_vector *x = gsl_vector_alloc (r+1);
            int res = solve_linear_system( cMat, bV, x );
            
            if ( res == 2 ) {
                printf( "The system for nu = %d has no solution!\n\n", nu );
                exit(1);
            }
            
            for ( i = 0; i < r; i++ )
                alphaV[i] = gsl_vector_get( x, i );
            
            // Error
            //printf("K = %.5e\n", gsl_vector_get( x, r ) );
            
            gsl_vector_free (x);
            
            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            //      Construct M_nu = alphaV_0 vepsMs[ind[0]] + ... + alphaV_{r-1} vepsMs[ind[r-1]]
            // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            
            gsl_vector_set_zero( M_nu );
            
            for ( i = 0; i < r; i++ ) {
                
                // M_nu = M_nu + alphaV_i vepsMs[ind[i]]
                gsl_vector_axpby( alphaV[i], vepsMs[indices_veps[i]], 1., M_nu );
            }
            
            // Free memory
            gsl_matrix_free( cMat );
            gsl_vector_free( bV );
            free( alphaV );
            free( indices_veps );
        }
        
        // * * * * * * * * * * *
        //    Error analysis
        // * * * * * * * * * * *
        
        if ( error_analysis == 1 ) {
            
            // Error of the solution just computed
            errors[nu] = compute_error ( n, Ms, vapsMs, M_nu );
        
            // If n_nu = 1, both the error and the norm are 0 and we don't need to normalize
            factor = 1;
            if ( error_relative_to_norm == 1 && n_nu > 1 ) {
                norm_pm = average_or_maximal_norm( n, Ms, vapsMs );
                
                // Normalization factor (the error will be divided by this number)
                factor = n * pow( norm_pm, 2 );
            }
            
            if ( fabs(factor) < 1.e-17 ) {
                printf( "Error: the normalization factor is 0.\n\n" );
                exit(1);
            }
            errors[nu] /= factor;
            
            // Collection of errors for a set of randomly generated vectors
            // We create nerror random vectors and compute the error in each case
            // The vectors are such that all their components lie in the range [ (1-a)/(n-1), a ]
            min = gsl_vector_min( M_nu );
            a = 1 - (n_nu-1) * min;
            if ( min > 0. )
                a = 1;
            else
                printf( "The observable vector has a negative component:\nmin = %.5f, a = %.5f\n", min, a );
            
            analyze_error( n, Ms, vapsMs, nerror, errors_random[nu], a, factor );
        }
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //      Store M_nu in the matrix that defines all the observables, mMat
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        // Only n_nu entries from index index_nu have to be modified
        for ( i = 0; i < n_nu; i++ )
            gsl_matrix_set ( mMat, nu, index_nu + i, gsl_vector_get( M_nu, i ) );
        
        // Free memory
        gsl_matrix_free( auxMat );
        gsl_vector_free( M_nu );
        
        for ( i = 0; i < n; i++ ) {
            gsl_matrix_free( Ms[i] );
            gsl_vector_free( vepsMs[i] );
        }
        free( Ms );
        free( vepsMs );
        free( vapsMs );
    }
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //        Compute the interaction matrix of the reduced system, deltaMat
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // deltaMat_{nu rho} = alpha_{nu rho} := Sum_{j in G_rho} (auxMat)_{nu j}
    // auxMat = mMat dMat
    
    auxMat = gsl_matrix_alloc ( n, N );
    gsl_blas_dgemm ( CblasNoTrans, CblasNoTrans, 1.0, mMat, dMat_gran, 0.0, auxMat );
    
    for ( nu = 0; nu < n; nu++ ) {
        
        for ( rho = 0; rho < n; rho++ ) {
            
            // deltaMat_{nu rho} = Sum_{j in G_rho} (auxMat)_{nu j}
            aux = 0;
            for ( j = 0; j < ns[rho]; j++ )
                aux += gsl_matrix_get( auxMat, nu, gsl_vector_int_get( group_indices, rho ) + j );
        
            gsl_matrix_set( deltaMat, nu, rho, aux );
        }
    }
    
    gsl_matrix_free( auxMat );
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //                        Define muMat and lambdaMat
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // Take lambdaMat = deltaMat
    gsl_matrix_memcpy( lambdaMat, deltaMat );
    
    // When there is no correction in the spectral method, muMat is taken to be equal to deltaMat
    if ( correction_red_syst != 1 && correction_red_syst != 2 )
        gsl_matrix_memcpy( muMat, deltaMat );
    
    else {
        
        // We take muMat such that it minimizes the error in the compatibility equations that
        // are related to the degree matrix:
        //
        //    muMat_{nu rho} = ( Mnu^T kMat_{nu rho} Mnu ) / || Mnu ||^2
        //                   = Sum_i ( (M_nu)_i^2 (k_{nu rho})_i ) / || Mnu ||^2,
        //
        //       where      k_{nu rho}                  is the vector of degrees of group nu (Gnu) coming from Grho,
        //                  kMat = diag( k_{nu rho} )   is the diagonal matrix of degrees from Grho to Gnu
        
        double norm2_nu, norm2_rho;
        
        for ( nu = 0; nu < n; nu++ ) {
            
            n_nu = ns[nu];
            index_nu = gsl_vector_int_get( group_indices, nu );
            
            // Vector M_nu
            M_nu = gsl_vector_alloc( n_nu );
            
            for ( i = 0; i < n_nu; i++ ) {
                aux = gsl_matrix_get( mMat, nu, i+index_nu );
                gsl_vector_set( M_nu, i, aux );
            }
            
            // Euclidean norm^2 of M_nu
            norm2_nu = pow( gsl_blas_dnrm2( M_nu ), 2 );
            
            for ( rho = 0; rho < n; rho++ ) {
                
                n_rho = ns[rho];
                index_rho = gsl_vector_int_get( group_indices, rho );
                
                // Vector M_rho
                M_rho = gsl_vector_alloc( n_rho );
                
                for ( i = 0; i < n_rho; i++ ) {
                    aux = gsl_matrix_get( mMat, rho, i+index_rho );
                    gsl_vector_set( M_rho, i, aux );
                }
                
                // Euclidean norm^2 of M_rho
                norm2_rho = pow( gsl_blas_dnrm2( M_rho ), 2 );
                
                // muMat_{nu rho} = Sum_i ( (M_nu)_i^2 k_{nu rho}_i ) / || Mnu ||^2
                aux = 0;
                for ( i = 0; i < n_nu; i++ ) {
                    
                    // Degree of node i coming from Grho
                    ki = 0;
                    
                    for ( j = 0; j < n_rho; j++ ) {
                        dij = gsl_matrix_get( dMat_gran, i + index_nu, j + index_rho );
                        ki += dij;
                    }
                    
                    aux += pow( gsl_vector_get( M_nu, i ), 2 ) * ki;
                }
                gsl_matrix_set( muMat, nu, rho, aux / norm2_nu );
                
                gsl_vector_free( M_rho );
            }
            
            gsl_vector_free( M_nu );
        }
    }
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //          Print the results of the error analysis
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    if ( error_analysis == 1 ) {
        
        FILE *fout;
        
        char nom[100];
        sprintf( nom, "errors_random_observables_n=%d.txt", n );
        fout = open_file( directory, nom, "w" );
        
        // Error of the observables
        mean_error = 0.;
        for ( nu = 0; nu < n; nu++ )
            mean_error += errors[nu];
        mean_error /= n;
        
        fprintf( fout, "# error for the observables:\n" );
        fprintf( fout, "# %13s %33s\n", "average error", "error associated to each group" );
        
        fprintf( fout , "%15.10f ", mean_error );
        for ( nu = 0; nu < n; nu++ )
            fprintf( fout, "%15.10f ", errors[nu] );
        fprintf( fout, "\n\n" );
        
        // Error for the random vectors
        fprintf( fout, "# errors for a collection of random vectors that add up to 1:\n" );
        fprintf( fout, "# %13s %33s\n", "average error", "error associated to each group" );
        
        for ( i = 0; i < nerror; i++ ) {
            
            mean_error = 0.;
            for ( nu = 0; nu < n; nu++ ) {
                mean_error += errors_random[nu][i];
            }
            mean_error /= n;
            
            fprintf( fout , "%15.10f ", mean_error );
            for ( nu = 0; nu < n; nu++ )
                fprintf( fout, "%15.10f ", errors_random[nu][i] );
            fprintf( fout, "\n" );
        }
        
        fclose( fout );
        
        free( errors );
        for ( i = 0; i < n; i++ )
            free( errors_random[i] );
        free( errors_random );
    }
    
    return;
}

// -------------
// Function F(x)
// -------------
double fF ( double x ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = -x;
            break;
            
        case 2:
            res = -x;
            break;
            
        case 3:
            res = B_fG + x * (1 - x / K_fG) * (x / C_fG - 1);
            break;
            
        default:
            res = 1-x;
            break;
    }
 
    return( res );
}

// --------------
// Function F'(x)
// --------------
double fF1 ( double x ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = -1;
            break;
            
        case 2:
            res = -1;
            break;
            
        case 3:
            res = - 3 * x * x / (C_fG * K_fG) + 2 * x * ( 1./C_fG + 1./K_fG ) - 1;
            break;
            
        default:
            res = -1;
            break;
    }
    return( res );
}

// ---------------
// Function G(x,y)
// ---------------
double fG ( double x, double y ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = pow( 1 + exp( -tau_fG * (y - mu_fG) ), -1 );
            break;
            
        case 2:
            res = gamma_fG * (1-x) * y;
            break;
            
        case 3:
            res = x * y / ( D_fG + E_fG * x + H_fG * y );
            break;
            
        default:
            res = y;
            break;
    }
    return( res );
}

// ----------------
// Function G1(x,y)
// ----------------
// Gives the derivative of G(x,y) resp. x
double fG1 ( double x, double y ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = 0;
            break;
        
        case 2:
            res = - gamma_fG * y;
            break;
            
        case 3:
            res = y * (D_fG + H_fG * y) / pow(D_fG + E_fG * x + H_fG * y, 2);
            break;
            
        default:
            res = 0;
            break;
    }
    return( res );
}

// -----------------
// Function G2(x, y)
// -----------------
// Gives the derivative of G(x,y) resp. y
double fG2 ( double x, double y ) {
    
    double res, aux;
    
    switch ( type_fFG ) {
        case 1:
            aux = exp( -tau_fG * (y-mu_fG) );
            res = tau_fG * aux * pow( 1 + aux, -2 );
            break;
            
        case 2:
            res = gamma_fG * (1-x);
            
        case 3:
            res = x * (D_fG + E_fG * x) / pow(D_fG + E_fG * x + H_fG * y, 2);
            break;
            
        default:
            res = 1;
            break;
    }
    return( res );
}

// -----------------
// Function G11(x,y)
// -----------------
// Gives the 2nd derivative of G(x,y) resp. x
double fG11 ( double x, double y ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = 0;
            break;
            
        case 2:
            res = 0;
            break;
            
        case 3:
            res = -2 * E_fG * y * (D_fG + H_fG * y) / pow(D_fG + E_fG * x + H_fG * y, 3);
            break;
            
        default:
            res = 0;
            break;
    }
    return( res );
}

// -----------------
// Function G12(x,y)
// -----------------
// Gives the derivative of G(x,y) resp. x and y
double fG12 ( double x, double y ) {
    
    double res;
    
    switch ( type_fFG ) {
        case 1:
            res = 0;
            break;
            
        case 2:
            res = - gamma_fG;
            break;
            
        case 3:
            res = ( D_fG * (D_fG + E_fG * x) + H_fG * ( D_fG + 2 * E_fG * x ) * y ) / pow( D_fG + E_fG * x + H_fG * y, 3 );
            break;
            
        default:
            res = 0;
            break;
    }
    return( res );
}

// ------------------
// Function G22(x, y)
// ------------------
// Gives the 2nd derivative of G(x,y) resp. y
double fG22 ( double x, double y ) {
    
    double res, aux;
    
    switch ( type_fFG ) {
        case 1:
            aux = exp( -tau_fG * (y-mu_fG) );
            res = 2 * pow( tau_fG * aux, 2 ) / pow( 1 + aux, 3 );
            res -= aux * pow( tau_fG, 2 ) / pow( 1 + aux, 2 );
            break;
            
        case 2:
            res = 0;
            
        case 3:
            res = - 2 * H_fG * x * (D_fG + E_fG * x) / pow(D_fG + E_fG * x + H_fG * y, 3);
            break;
            
        default:
            res = 0;
            break;
    }
    return( res );
}


// -------------------------------------------------------------------------------------------
// Derivative of the i-th function tau * dxi / dt with respect to variable xk (evaluated at y)
// for the model WITHOUT CORRECTION
// -------------------------------------------------------------------------------------------

double derivative_xi_resp_xk ( struct params_all *params, int i, int k, const double y[] ) {
    
    int j;
    double res, d;
    
    int N = ((struct params_all *) params)->N;
    gsl_matrix *dMat = ((struct params_all *) params)->dMat;
    
    d = gsl_matrix_get( dMat, i, k );               // d_{ik}
    res = d * fG2( y[i], y[k] );

    if ( i == k ) {
        
        res += fF1( y[i] );
        
        for ( j = 0; j < N; j++ ) {
            
            d = gsl_matrix_get( dMat, i, j );         // d_{ij}
            res += d * fG1( y[i], y[j] );
        }
    }
    
    return( res );
}

// -------------------------------------------------------------------------------------------
// Derivative of the i-th function tau * dxi / dt with respect to variable xk (evaluated at y)
// for the model WITH CORRECTION 1
// -------------------------------------------------------------------------------------------

double derivative_xi_resp_xk_corr1 ( struct params_all *params, int i, int k, const double y[] ) {
    
    int j;
    double res, d, lambda, mu, a, b;
    
    int N = ((struct params_all *) params)->N;
    gsl_matrix *dMat = ((struct params_all *) params)->dMat;
    gsl_matrix *muMat = ((struct params_all *) params)->muMat;
    gsl_matrix *lambdaMat = ((struct params_all *) params)->lambdaMat;

    d = gsl_matrix_get( dMat, i, k );               // d_{ik}
    lambda = gsl_matrix_get( lambdaMat, i, k );     // lambda_{ik}
    mu = gsl_matrix_get( muMat, i, k );             // mu_{ik}
    
    if ( fabs(d) > 1e-10 ) {
        a = mu / d;
        b = lambda / d;
    }
    else { // This correction cannot be defined (it is not possible to cancel the derivative terms)
        printf( "It is not possible to implement the correction that cancels the derivatives because d = 0.\n\n" );
        exit(1);
    }
    
    res = lambda * fG2( a * y[i], b * y[k] );
    
    if ( i == k ) {
        
        res += fF1( y[i] );
        
        for ( j = 0; j < N; j++ ) {
            
            d = gsl_matrix_get( dMat, i, j );               // d_{ij}
            lambda = gsl_matrix_get( lambdaMat, i, j );     // lambda_{ij}
            mu = gsl_matrix_get( muMat, i, j );             // mu_{ij}
            
            if ( fabs(d) > 1e-10 ) {
                a = mu / d;
                b = lambda / d;
            }
            else { // This correction cannot be defined (it is not possible to cancel the derivative terms)
                printf( "It is not possible to implement the correction that cancels the derivatives because d = 0.\n\n" );
                exit(1);
            }
            
            res += mu * fG1( a * y[i], b * y[j] );
        }
    }
    
    return( res );
}

// -------------------------------------------------------------------------------------------
// Derivative of the i-th function tau * dxi / dt with respect to variable xk (evaluated at y)
// for the model WITH CORRECTION 2
// -------------------------------------------------------------------------------------------

double derivative_xi_resp_xk_corr2 ( struct params_all *params, int i, int k, const double y[] ) {
    
    int j;
    double res, d, lambda, mu, a, b;
    
    int N = ((struct params_all *) params)->N;
    gsl_matrix *dMat = ((struct params_all *) params)->dMat;
    gsl_matrix *muMat = ((struct params_all *) params)->muMat;
    gsl_matrix *lambdaMat = ((struct params_all *) params)->lambdaMat;
    
    res = derivative_xi_resp_xk ( params, i, k, y );
    
    // Afegim els termes correctors
    d = gsl_matrix_get( dMat, i, k );               // d_{ik}
    mu = gsl_matrix_get( muMat, i, k );             // mu_{ik}
    lambda = gsl_matrix_get( lambdaMat, i, k );     // lambda_{ik}
    
    a = mu - d;
    b = lambda - d;
    
    res += a * fG12( y[i], y[k] ) * y[i];
    res += b * ( fG22( y[i], y[k] ) * y[k] + fG2( y[i], y[k] ) );
    
    if ( i == k ) {
        
        for ( j = 0; j < N; j++ ) {
            
            d = gsl_matrix_get( dMat, i, j );               // d_{ij}
            mu = gsl_matrix_get( muMat, i, j );             // mu_{ij}
            lambda = gsl_matrix_get( lambdaMat, i, j );     // lambda_{ij}
            
            a = mu - d;
            b = lambda - d;
            
            res += a * ( fG11( y[i], y[j] ) * y[i] + fG1( y[i], y[j] ) );
            res += b * fG12( y[i], y[j] ) * y[j];
        }
    }
    
    return( res );
}

// --------------------------------------------------------------------------------------
// Function that defines the temporal derivatives of the complete system (evaluated at y)
// --------------------------------------------------------------------------------------
int func ( double t, const double y[], double f[], void *params ) {
    
    (void)(t); /* avoid unused parameter warning */
    int i, j, N, reduced_system, correction_red_syst;
    gsl_matrix *dMat;
    double tau, d, lambda, mu, a, b;
    
    N = ((struct params_all *) params)->N;
    tau = ((struct params_all *) params)->tau;
    dMat = ((struct params_all *) params)->dMat;
    gsl_matrix *muMat = ((struct params_all *) params)->muMat;
    gsl_matrix *lambdaMat = ((struct params_all *) params)->lambdaMat;
    reduced_system = ((struct params_all *) params)->reduced_system;
    correction_red_syst = ((struct params_all *) params)->correction_red_syst;
    
    // + + + + + + + + + + + + + + + +
    // REDUCED SYSTEM WITH CORRECTION
    // + + + + + + + + + + + + + + + +
    
    if ( reduced_system == 1 && ( correction_red_syst == 1 || correction_red_syst == 2 ) ) {
        
        // Correction 1: the 1st order terms of the reduced system are cancelled
        // and the Taylor points are modified (like in Laurence et al., Thibeault et al.)
        if ( correction_red_syst == 1 ) {
            
            // Define the temporal derivatives of the activity variables
            for ( i = 0; i < N; i++ ) {
                
                f[i] = fF( y[i] );
                
                for ( j = 0; j < N; j++ ) {
                    
                    d = gsl_matrix_get( dMat, i, j );               // d_{ij}
                    mu = gsl_matrix_get( muMat, i, j );             // mu_{ij}
                    lambda = gsl_matrix_get( lambdaMat, i, j );     // lambda_{ij}
                    
                    if ( fabs(d) > 1e-10 ) {
                        a = mu / d;
                        b = lambda / d;
                    }
                    else { // This correction cannot be defined (it is not possible to cancel the derivative terms)
                        printf( "It is not possible to implement the correction that cancels the derivatives because d = 0.\n\n" );
                        exit(1);
                    }
                    
                    f[i] += d * fG( a*y[i], b*y[j] );
                }
                
                f[i] /= tau;
            }
        }
       
        // Correction 2: the Taylor points are the observables and we consider the 1st order terms
        if ( correction_red_syst == 2 ) {
            
            // Define the temporal derivatives of the activity variables
            for ( i = 0; i < N; i++ ) {
                
                f[i] = fF( y[i] );
                
                for ( j = 0; j < N; j++ ) {
                    
                    d = gsl_matrix_get( dMat, i, j );               // d_{ij}
                    mu = gsl_matrix_get( muMat, i, j );             // mu_{ij}
                    lambda = gsl_matrix_get( lambdaMat, i, j );     // lambda_{ij}
                    
                    f[i] += d * fG( y[i], y[j] ); // Linear terms
                    
                    // Add the 1st order terms
                    f[i] += fG1( y[i], y[j] ) * (mu - d) * y[i];
                    f[i] += fG2( y[i], y[j] ) * (lambda - d) * y[j];
                }
                
                f[i] /= tau;
            }
        }
    }

    // + + + + + + + + + + + + + + + + + + + + + + + + + + +
    // ORIGINAL SYSTEM OR REDUCED SYSTEM WITHOUT CORRECTION
    // + + + + + + + + + + + + + + + + + + + + + + + + + + +
    
    else {
        
        // Define the temporal derivatives of the activity variables
        for ( i = 0; i < N; i++ ) {
            
            f[i] = fF( y[i] );
            
            for ( j = 0; j < N; j++ ) {
                
                d = gsl_matrix_get( dMat, i, j );               // d_{ij}
                f[i] += d * fG( y[i], y[j] );
            }
            
            f[i] /= tau;
        }
    }

    return GSL_SUCCESS;
}

// ----------------
// Jacobian of func
// ----------------
// The system has N variables, so the Jacobian has dimension N^2

int jac ( double t, const double y[], double *dfdy, double dfdt[], void *params ) {
    
    int i, j, reduced_system, correction_red_syst;
    double der;
    (void)(t); /* avoid unused parameter warning */
    
    int N = ((struct params_all *) params)->N;
    double tau = ((struct params_all *) params)->tau;
    reduced_system = ((struct params_all *) params)->reduced_system;
    correction_red_syst = ((struct params_all *) params)->correction_red_syst;
    
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, N, N);
    gsl_matrix * m = &dfdy_mat.matrix;
    
    // + + + + + + + + + + + + + + + +
    // REDUCED SYSTEM WITH CORRECTION
    // + + + + + + + + + + + + + + + +
    
    if ( reduced_system == 1 && ( correction_red_syst == 1 || correction_red_syst == 2 ) ) {
        
        // Correction 1: the 1st order terms of the reduced system are cancelled
        // and the Taylor points are modified (like in Laurence et al., Thibeault et al.)
        if ( correction_red_syst == 1 ) {
            
            // Run through rows and columns of matrix m
            for ( i = 0; i < N; i++ ) { // rows 0-(n-1) -> derivatives of d xi / dt
                
                for ( j = 0; j < N; j++ ) { // columns 0-(n-1) -> derivatives w.r. to xj
                    
                    der = 1./tau * derivative_xi_resp_xk_corr1 ( params, i, j, y );
                    gsl_matrix_set ( m, i, j, der );
                }
            }
        }
        
        // Correction 2: the Taylor points are the observables and we consider the 1st order terms
        if ( correction_red_syst == 2 ) {
            
            // Run through rows and columns of matrix m
            for ( i = 0; i < N; i++ ) { // rows 0-(n-1) -> derivatives of d xi / dt
                
                for ( j = 0; j < N; j++ ) { // columns 0-(n-1) -> derivatives w.r. to xj
                    
                    der = 1./tau * derivative_xi_resp_xk_corr2 ( params, i, j, y );
                    gsl_matrix_set ( m, i, j, der );
                }
            }
        }
    }
    
    // + + + + + + + + + + + + + + + + + + + + + + + + + + +
    // ORIGINAL SYSTEM OR REDUCED SYSTEM WITHOUT CORRECTION
    // + + + + + + + + + + + + + + + + + + + + + + + + + + +
    
    else {
        
        // Run through rows and columns of matrix m
        for ( i = 0; i < N; i++ ) { // rows 0-(n-1) -> derivatives of d xi / dt
            
            for ( j = 0; j < N; j++ ) { // columns 0-(n-1) -> derivatives w.r. to xj
                
                der = 1./tau * derivative_xi_resp_xk ( params, i, j, y );
                gsl_matrix_set ( m, i, j, der );
            }
        }
    }
    
    // The time derivatives are all 0 (autonomous system)
    for ( i = 0; i < N; i++ )
        dfdt[i] = 0.;
    
    return GSL_SUCCESS;
}

// ------------------------
// Print the system's state
// ------------------------
int print_state (size_t iter, gsl_multiroot_fsolver * s) {
    
    printf ("iter = %zu, x = (%.5f %.5f), "
            "f(x) = (%.5e %.5e)\n",
            iter,
            gsl_vector_get (s->x, 0),
            gsl_vector_get (s->x, 1),
            gsl_vector_get (s->f, 0),
            gsl_vector_get (s->f, 1));
    
    return (0);
}

// -----------------
// Initial condition
// -----------------
void initial_condition( int N, double x_ini0, double x_ini1, double *y ) {
    
    int i;
    
    if ( x_ini0 > x_ini1 )
        printf( "Error in the initial condition introduced. We set all the initial activities at %.3f.\n\n", x_ini0 );
    
    for ( i = 0; i < N; i++ )
        y[i] = x_ini0 + RANDOM * (x_ini1 - x_ini0);
    
    return;
}

// -----------------------------
// Perturb the initial condition
// -----------------------------
void perturb_initial_condition( int N, double range, double *y ) {
    
    int i;
    
    for ( i = 0; i < N; i++ )
        y[i] += RANDOM * range;

    return;
}

// ----------
// Print step
// ----------
void print_step( FILE *fout_act, int N, double t, const double y[] ) {
    
    int i;
    
    fprintf( fout_act, "\n%10.5f ", t );

    for ( i = 0; i < N; i++ )
        fprintf( fout_act, "%10.5f ", y[i] );
    
    return;
}

// -------------------------------------------------------------------------------
// Print the final state (equilibrium) of a simulation for a given parameter value
// -------------------------------------------------------------------------------
// param: parameter
// n: number of groups in the reduction
// m: number of variables
// sizes: sizes of groups
// average_only:  1  -> print the average state of nodes only
//              != 1 -> print the average state + the state of each node
// If the function is called to print the microscopic equilibrium, then sizes[i] = 1 for all i or sizes = NULL

void print_equilibrium( FILE *fout, double param, int n, int m, const double y[], int *sizes, int average_only ) {
    
    int i;
    double average; // observables' average
    
    fprintf( fout, "%12.7f ", param );
    
    // Average activity of nodes, weighted by group sizes or not (depending on whether sizes == NULL )
    average = observable_average ( n, y, sizes );
    fprintf( fout, "%17.7f ", average );
    
    // Node activity
    if ( average_only != 1 ) {
        
        for ( i = 0; i < m; i++ )
            fprintf( fout, "%17.7f ", y[i] );
    }
    
    fprintf( fout, "\n" );
    
    return;
}

// --------------------------------
// Average of y weighted by "sizes"
// --------------------------------
// If sizes == NULL, the non-weighted average is computed

double observable_average ( int n, const double y[], int *sizes ) {
    
    int i, suma_sizes = 0;
    double average = 0.;
    
    if ( sizes != NULL ) {
        
        for ( i = 0; i < n; i++ )
            suma_sizes += sizes[i];
        
        for ( i = 0; i < n; i++ )
            average += sizes[i] * y[i];
        average /= suma_sizes;
    }
    
    else {
        for ( i = 0; i < n; i++ )
            average += y[i];
        average /= n;
    }
    
    return ( average );
}

// ----------------------------------------------------------------
// Compute the observables from the microscopic state of the system
// ----------------------------------------------------------------
void compute_observables ( struct params_all *params, double *y, double *yMacro ) {
    
    int i;
    gsl_vector *xV, *xMacroV;
    
    int N = params->N;
    int n = params->n;
    gsl_matrix *mMat = params->mMat;
    
    xV = gsl_vector_alloc( N );
    xMacroV = gsl_vector_calloc( n );
    
    for ( i = 0; i < N; i++ )
        gsl_vector_set( xV, i, y[i] );
    
    // xMacroV = mMat xV
    gsl_blas_dgemv( CblasNoTrans, 1.0, mMat, xV, 0.0, xMacroV );
    
    for ( i = 0; i < n; i++ )
        yMacro[i] = gsl_vector_get( xMacroV, i );
    
    gsl_vector_free( xV );
    gsl_vector_free( xMacroV );
    
    return;
}

// ------------------------------------------
// Integrate the system (original or reduced)
// ------------------------------------------

// "y" is the system's state. At the beginning it is the initial condition.
// "yMacro" is the observables' state when we integrate the original system. It doesn't need to be initialized

// The trajectories are printed in files. The number of steps printed is given by the parameter "steps".

int integrate_system ( struct params_all *params, double *y, double *yMacro ) {
    
    int i, status, step;
    double ti, t = 0.0, *y0, dist, tol;
    gsl_vector *xMacroV;
    char nom[300];
    FILE *fout, *fout_macro;
    gsl_odeiv2_driver * d;
    
    // ............................
    //         Parameters
    // ............................
    
    // macro = 1  -> we integrate the reduced (macro) system
    //        !1  -> we integrate the original (micro) system
    int macro = params->reduced_system;
    
    int N = ((struct params_all *) params)->N;  // number of variables, either if we are integrating the micro or the macro system
    int n = ((struct params_all *) params)->n; // number of observables if we integrate the micro system
    double t_sim = ((struct params_all *) params)->t_sim;
    int steps = ((struct params_all *) params)->steps;
    char *directory = ((struct params_all *) params)->directory;
    
    gsl_odeiv2_system sys = {func, jac, N, params};
    d = gsl_odeiv2_driver_alloc_y_new ( &sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0 );
    
    y0 = allocate_vector_double ( N, "y0" );
    for ( i = 0; i < N; i++ )
        y0[i] = -100;
    
    tol = 1.e-12;
    
    // Output files with the trajectories
    if ( macro != 1 ) {
        
        // File with the micro dynamics
        sprintf( nom, "dynamics_micro.txt" );
        fout = open_file( directory, nom, "w" );
        
        // File with the exact macro dynamics
        sprintf( nom, "dynamics_macro_from_micro.txt" );
        fout_macro = open_file( directory, nom, "w" );
        
        fprintf( fout_macro, "# %8s ", "time" );
        for ( i = 0; i < n; i++ )
            fprintf( fout_macro, "%8s%-2d ", "x", i+1 );
    }
    
    else {
        // File with the reduced macro dynamics
        sprintf( nom, "dynamics_macro_simulated.txt" );
        fout = open_file( directory, nom, "w" );
    }
    
    fprintf( fout, "# %8s ", "time" );
    for ( i = 0; i < N; i++ )
        fprintf( fout, "%8s%-2d ", "x", i+1 );
    
    // Print the system's state at t=0
    print_step( fout, N, t, y );
    
    // If we integrate the micro system, compute and print the observables at t=0
    if ( macro != 1 ) {
        
        xMacroV = gsl_vector_alloc( n );
        compute_observables ( params, y, yMacro );
        print_step( fout_macro, n, t, yMacro );
    }
    
    // Integrate
    for ( step = 1; step <= steps; step++ ) {
        
        ti = step * t_sim / steps;
        status = gsl_odeiv2_driver_apply ( d, &t, ti, y );
        
        if (status != GSL_SUCCESS) {
            
            printf ("error, return value=%d\n", status);
            exit(1);
        }
        
        // If the current step is very close to the previous step, we stop integrating
        dist = euc_dist_vectors( N, y0, y ) / N;
        if ( dist < tol ) {
            printf( "I stop integrating because I have converged!\n" );
            break;
        }
        else
            copy_vector_double( N, y, y0 );
        
        // Print the current system's state
        print_step( fout, N, t, y );
        
        // If we integrate the micro system, compute and print the observables
        if ( macro != 1 ) {
            
            compute_observables ( params, y, yMacro );
            print_step( fout_macro, n, t, yMacro );
        }
    }
    
    gsl_odeiv2_driver_free (d);
    free( y0 );
    
    fclose( fout );
    
    if ( macro != 1 ) {
        
        fclose( fout_macro );
        gsl_vector_free( xMacroV );
    }
    
    return 0;
}

// -----------------------------
// Print the system's parameters
// -----------------------------

void print_params ( struct params_all params ) {
    
    FILE *fout;
    
    int N = params.N;
    int n = params.n;
    double tau = params.tau;
    double t_sim = params.t_sim;
    int steps = params.steps;
    char *directory = params.directory;
    double d0 = params.d0;
    
    fout = open_file( directory, "parameters.txt", "w" );
    
    fprintf( fout, "N = %d\nn = %d\ntau = %.3f\nd0 = %.5e\n\n", N, n, tau, d0 );
    
    switch ( type_fFG ) {
            
        case 1: // WC
            fprintf( fout, "Neuronal dynamics\ntau_g = %.5f\nmu = %.5f\n\n", tau_fG, mu_fG );
            break;
        
        case 2: // SIS
            fprintf( fout, "SIS dynamics\ngamma = %.5f\n\n", gamma_fG );
            break;
            
        case 3: // Ecology
            fprintf( fout, "Ecological dynamics\nB = %.5f\nC = %.5f\nK = %.5f\nD = %.5f\nE = %.5f\nH = %.5f\n\n", B_fG, C_fG, K_fG, D_fG, E_fG, H_fG );
            break;
            
        default:
            fprintf( fout, "Default dynamics\n\n" );
            break;
    }
    
    fprintf( fout, "t_sim = %.3f\nsteps = %d\n\n", t_sim, steps );
    
    fclose( fout );
    return;
}

// -------------------------------
// Compute the bifurcation diagram
// -------------------------------

// At each point in the bif. diagram, the original interaction matrix is multiplied by a factor d.
// The collection of d factors is specified by the vector d_v inside params_bd.
//
// For each parameter d, the system is integrated to equilibrium and the result is written in a file.
//
// integrate_micro: def -> Both the micro and the macro systems are integrated.
//                   0  -> Only the macro system is integrated.

void bifurcation_diagram ( struct params_all *params, struct params_cond_ini *params_ini, struct params_bif_diag params_bd, char *name, char *name_red, int error_analysis, int print_parameters, int integrate_micro ) {
 
    int i, j, k, status;
    double d, dini, alpha, factor_d, range;
    double x_ini0, x_ini1;
    double *y, *yMacro, *yMacroRed, *yMacroRed0;
    char dir[300], subdir[300], name_aux[300];
    FILE *f_output, *f_outputRed, *f_deltaMat, *f_muMat, *f_micro;
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                                  PARAMETERS                                 //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    factor_d = 1;
    
    sprintf( dir, "%s", params->directory );
    
    int N = ((struct params_all *) params)->N;
    int n = ((struct params_all *) params)->n;
    double d0 = params->d0;
    int *sizes = params->ns;
    int reduction_method = params->reduction_method;
    
    double *d_v = params_bd.d_v;
    double *alpha_v = params_bd.alpha_v;
    int nd = params_bd.nd;
    
    gsl_matrix *dMat_or = params->dMat_or;
    gsl_matrix *dMat = gsl_matrix_calloc( N, N );
    gsl_matrix *lambdaMat = gsl_matrix_calloc( n, n );
    gsl_matrix *muMat = gsl_matrix_calloc( n, n );
    gsl_matrix *deltaMat = gsl_matrix_calloc( n, n );
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                                 OUTPUT FILES                                //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    if ( integrate_micro != 0 ) {
        f_output = open_file( dir, name, "w" );
        fprintf( f_output, "# %10s %17s %30s\n", "Param.", "Average node obs.", "Obs. from the micro. system" );
        f_micro = open_file( dir, "equilibrium_micro.txt", "w" );
    }
    
    f_outputRed = open_file( dir, name_red, "w" );
    fprintf( f_outputRed, "# %10s %17s %30s\n", "Param.", "Average node obs.", "Obs. from the reduced system" );
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                       VARIABLES AND INITIAL CONDITION                       //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    // Variables micro and macro
    y = allocate_vector_double ( N, "y" );
    yMacro = allocate_vector_double ( n, "yMacro" );
    yMacroRed = allocate_vector_double ( n, "yMacroRed" );
    yMacroRed0 = allocate_vector_double ( n, "yMacroRed0" );
    
    // Initial condition micro
    x_ini0 = params_ini->x_ini0;
    x_ini1 = params_ini->x_ini1;
    initial_condition( N, x_ini0, x_ini1, y );

    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                            BIFURCATION DIAGRAM                              //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    dini = d_v[0] / N;
    
    for ( i = 0; i < nd; i++ ) {
        
        d = d_v[i] / N;
        
        if ( fabs(dini) < 1.e-14 ) {
            printf( "dini cannot be 0.\n\n" );
            exit(1);
        }
        
        // Ratio between the current d and that of the previous point in the bif. diagram.
        // It is used to easily compute the properties of the reduction from those of the previous case.
        factor_d = d / dini;
        
        dini = d;
        
        if ( i != 0 )
            error_analysis = 0;
        
        printf( "\n---------------------------\n   Case d = %.4f\n---------------------------\n", d );
        
        // Subdirectory
        sprintf( subdir, "%s/d=%.2f", dir, d );
        status = mkdir( subdir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
        sprintf( params->directory, "%s", subdir );
        
        params->N = N;
        params->dMat = dMat;
        params->deltaMat = deltaMat;
        params->lambdaMat = lambdaMat;
        params->muMat = muMat;
        
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        //             MICROSCOPIC INTERACTION MATRIX dMat = (d_ij)_i,j
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        
        // Multiply all the elements of the original matrix by factor d
        rescale_dMat ( d, d0*d, dMat, dMat_or );
        
        
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        //     COMPUTE THE OBSERVABLES AND THE PARAMETERS OF THE REDUCED SYSTEM
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        
        //   * Homogeneous (Naive) reduction:
        //        - The vectors that define the observables are homogeneous over every group
        //        - The parameters that define the reduced system are the average in-degrees
        //          of every group from all the other groups
        //
        //   * Degree-based reduction (n=1 only):
        //        - The observable vector is (kout_1, ..., kout_N) / <k>, where <k> = Sum_i kout_i
        //        - The reduced system's parameter is <kout kin> / <k>, where <kout kin> = Sum_i ( kout_i * kin_i )
        //
        //   * Spectral (eigenval) reduction:
        //        - The vectors that define the observables and the parameters of the reduced system
        //          are computed by solving the compatibility equations
        //        - If n=1:
        //          á the vector is the leading eigenvector of dMat
        //          á the parameter is the leading eigenvalue of dMat
        //
        // OBSERVATION:
        //      When at each step in the bif. diagram the interaction matrix is multiplied by a factor "factor_d",
        //      then the observables are always the same and the only thing that changes is that the reduced
        //      system's parameters are multiplied by "factor_d".
        // ERGO:
        //      We only need to compute these parameters ONCE.
        // BUT:
        //      If we chose to define the bif. diagram in a different way, this could no longer be the case.
        
        // First time: compute the observables and the reduced system's parameters
        if ( i == 0 ) {
            
            switch ( reduction_method ) {
                case 1: // Homogeneous reduction
                    compute_parameters_homogeneous ( params );
                    break;
                    
                case 2: // Degree-based reduction
                    params->n = 1;
                    compute_parameters_degree ( params );
                    break;
                    
                default: // Spectral reduction
                    solve_compatibility_eqs ( params, error_analysis );
                    break;
            }
        }
        
        // Reescale the reduced system's parameters
        else {
            gsl_matrix_scale( deltaMat, factor_d );
            gsl_matrix_scale( lambdaMat, factor_d );
            gsl_matrix_scale( muMat, factor_d );
        }
        
        // Print the parameters of the reduced system
        if ( print_parameters == 1 ) {
            
            // Print the observable vectors
            // (when the method is the Homogeneous, we don't print them because they are always the homogeneous vectors)
            if ( reduction_method != 1 ) {
                sprintf( name_aux, "observable_vectors.txt" );
                print_vectors_observables( params, name_aux, subdir );
            }
            
            // Print the reduced interaction matrix
            sprintf( name_aux, "delta_matrix.txt" );
            f_deltaMat = open_file( subdir, name_aux, "w" );
            print_matrix_gsl( f_deltaMat, deltaMat );
            fclose( f_deltaMat );
            
            // Print mu matrix
            sprintf( name_aux, "mu_matrix.txt" );
            f_muMat = open_file( subdir, name_aux, "w" );
            print_matrix_gsl( f_muMat, muMat );
            fclose( f_muMat );
        
            // Print lambda matrix
            sprintf( name_aux, "lambda_matrix.txt" );
            f_muMat = open_file( subdir, name_aux, "w" );
            print_matrix_gsl( f_muMat, lambdaMat );
            fclose( f_muMat );
        }
        
        // We denote by alpha the parameter that we use to identify each step in the bif. diagram
        // (calligraphic K in the plots)
        // We take alpha as the average in-degree of the reduced system (weighted by group sizes)
        // This makes alpha be in a given range independently of the size of the reduced system (n)
        // This is so because deltaMat[j][k] is proportional to the size of group k
        alpha = 0;
        for ( j = 0; j < n; j++ ) {
            for ( k = 0; k < n; k++ )
                alpha += sizes[j] * gsl_matrix_get( deltaMat, j, k );
        }
        
        alpha /= N;
        alpha_v[i] = alpha;
        //printf( "Parameter of the reduced system: %.3f\n\n", alpha );
        
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        //                      MACROSCOPIC INITIAL CONDITION
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        
        // The first time, the macro initial condition is the one corresponding to the micro initial condition
        // Then, the macro initial condition is the final state of the reduced system in the previous simulation
        if ( i == 0 )
            compute_observables ( params, y, yMacroRed );
        
        // In the SIS dynamics, we perturb the initial condition because the system has problems to escape
        // the 0 state for some parameter values
        if ( type_fFG == 2 ) {
            range = 0.1;
            perturb_initial_condition( N, range, y );
            perturb_initial_condition( n, range, yMacroRed );
        }
        
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        //                              INTEGRATE
        // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
        
        // =====================================
        // Integrate the MICRO (original) system
        // =====================================
        
        if ( integrate_micro != 0 ) {
            
            // Now lambdaMat and muMat have to be equal to dMat
            params->N = N;
            params->dMat = dMat;
            params->lambdaMat = dMat;
            params->muMat = dMat;
            params->reduced_system = 0;
            
            printf( "Integrating the ORIGINAL system...\n\n" );
            
            status = integrate_system( params, y, yMacro );
            
            // Print the microscopic equilibrium
            print_equilibrium( f_micro, alpha, N, N, y, NULL, 0 );
            
            // Print the exact observables from this equilibrium
            print_equilibrium( f_output, alpha, n, n, yMacro, sizes, 0 );
        }
        
        // ====================================
        // Integrate the MACRO (reduced) system
        // ====================================
        
        // Modify the key parameters to integrate the reduced system using the same function
        params->N = n;
        params->dMat = deltaMat;
        params->lambdaMat = lambdaMat;
        params->muMat = muMat;
        params->reduced_system = 1;
        
        printf( "Integrating the REDUCED system...\n\n" );
        
        status = integrate_system( params, yMacroRed, NULL );
        
        // Print the observables at equilibrium computed from the reduced system
        print_equilibrium( f_outputRed, alpha, n, n, yMacroRed, sizes, 0 );
    }
    
    // Restore parameters
    sprintf( params->directory, "%s", dir );
    sprintf( params->subdirectory, "%s", subdir );
    params->N = N;
    
    // Free memory
    gsl_matrix_free( dMat );
    gsl_matrix_free( lambdaMat );
    gsl_matrix_free( muMat );
    gsl_matrix_free( deltaMat );
    
    free( y );
    free( yMacro );
    free( yMacroRed );
    free( yMacroRed0 );
    
    if ( integrate_micro != 0 )
        fclose( f_output );
    fclose( f_outputRed );
    
    if ( integrate_micro != 0 )
        fclose( f_micro );
    
    return;
}

// -----------------------------------
// Read the bifurcation diagram's data
// -----------------------------------

// We read only the 2nd column, which gives the value of the average observable
// If data == NULL, we only read the data of the reduced system

void read_data_bif_diagram ( gsl_vector *data, gsl_vector *data_red, char *name, char *name_red, char *directory ) {
    
    int i, ll;
    double aux, datum;
    char chain[300];
    FILE *f, *f_red;
    
    ll = (int) data_red->size;
    
    if ( data != NULL && (data->size != ll) ) {
        printf( "Error: data and data_red must have the same dimension.\n\n" );
        exit(1);
    }
    
    // Read the reduced system's data
    f_red = open_file( directory, name_red, "r" );
    fgets( chain, 300, f_red );
    
    for ( i = 0; i < ll; i++ ) {
        
        fscanf( f_red, "%lf", &aux );
        fscanf( f_red, "%lf", &datum );
        fgets( chain, 300, f_red );
        gsl_vector_set( data_red, i, datum );
    }
    fclose( f_red );
    
    // Read the original system's data
    if ( data != NULL ) {
        
        f = open_file( directory, name, "r" );
        fgets( chain, 300, f );
        
        for ( i = 0; i < ll; i++ ) {
            
            fscanf( f, "%lf", &aux );
            fscanf( f, "%lf", &datum );
            fgets( chain, 300, f );
            gsl_vector_set( data, i, datum );
        }
        fclose( f );
    }
    
    return;
}

// -------------------------------------
// Read the bif. diagram's COMPLETE data
// -------------------------------------
// We read the data corresponding to every single node, for each parameter value

void read_complete_data_bif_diag ( gsl_matrix *data, char *name, char *directory ) {
    
    int i, j, ll, n;
    double aux, datum;
    FILE *f;
    
    ll = (int) data->size1;
    n = (int) data->size2;
    
    f = open_file( directory, name, "r" );
    
    for ( i = 0; i < ll; i++ ) {
        
        fscanf( f, "%lf", &aux );
        fscanf( f, "%lf", &aux );
        
        for ( j = 0; j < n; j++ ) {
            fscanf( f, "%lf", &datum );
            
            gsl_matrix_set( data, i, j, datum );
        }
    }
    fclose( f );
    
    return;
}

// ---------------------------------------------------------------------------
// Compute the bif. diagram's data when the node membership has been perturbed
// ---------------------------------------------------------------------------

// data_or:  matrix with the original data
//           (for every parameter in the diagram, as many data as nodes (N data)
//
// data:     final bif. diagram's data for the perturbed case
//           (for every parameter in the diagram, as many data as observables (n data)
//
// map: node permutation corresponding to the perturbation
//      map[i]: original index of the node that finally occupies position i

void compute_data_bif_diag ( gsl_matrix *data_or, gsl_matrix *data, int *map, struct params_all *params ) {
 
    int i, i0, i_alpha, N, n, nalpha;
    int *sizes;
    double *y, *yMacro;
    
    // Number of steps or parameters in the diagram
    nalpha = (int) data_or->size1;
    
    N = params->N;
    n = params->n;
    sizes = params->ns;
    
    if ( data_or->size2 != N ) {
        printf( "There is a dimension problem: data->size2 should be %d and it is %d.\n\n", N, (int) data_or->size2 );
        exit(1);
    }
    
    y = allocate_vector_double( N, "y" );
    yMacro = allocate_vector_double( n, "yMacro" );
    
    // Run over the parameter index in the diagram
    for ( i_alpha = 0; i_alpha < nalpha; i_alpha++ ) {
     
        // Compute the microscopic state according to the new node ordering
        for ( i = 0; i < N; i++ ) {
            
            // "i0" is the old index that corresponds to the current node "i"
            i0 = map[i];
            
            // State of the i-th node according to the new ordering
            y[i] = gsl_matrix_get( data_or, i_alpha, i0 );
        }
        
        // Compute the value of the observables
        compute_observables ( params, y, yMacro );
        
        // Introduce the observables in the data vector
        for ( i = 0; i < n; i++ )
            gsl_matrix_set( data, i_alpha, i, yMacro[i] );
    }
    
    free( y );
    free( yMacro );
    
    return;
}

// -------------------------------------------------------------
// Error (root-mean-square error, RMSE) between two data vectors
// -------------------------------------------------------------

double RMSE ( gsl_vector *u, gsl_vector *v ) {
    
    int i, n;
    double error, aux;
    
    n = (int) u->size;
    
    if ( v->size != n ) {
        printf( "Error: u and v must have the same dimension.\n\n" );
        exit(1);
    }
    
    error = 0.;
    
    for ( i = 0; i < n; i++ ) {
     
        aux = gsl_vector_get( u, i ) - gsl_vector_get( v, i );
        aux = pow( aux, 2 );
        error += aux;
    }
    
    // Normalize
    error = sqrt( error / n );
    
    return ( error );
}

// -------------------------------------------
// Define the bifurcation diagram's parameters
// -------------------------------------------

// The bif. diag. is computed by varying the overall strength the connections.
// At each step, the connections are multiplied by a constant factor d / N with d in [d_ini, d_fin]
// The number of ds used is nd and the collection of ds is stored in d_v.

void define_bif_diag_parameters ( struct params_interaction_matrix *params, struct params_bif_diag *params_bd ) {
    
    int i, nd;
    double d_ini, d_fin, dd, *d_v, *alpha_v;
    
    double rho = params->rho;
    int real_network = params->real_network;
    int n_blocks = params->n_blocks;

    // Define d_ini, d_fin according to the network type and dynamics
    d_ini = 50;
    d_fin = 150;
    
    switch ( type_fFG ) {
            
        case 1: // WC
            
            if ( fabs(rho) < 0.2 ) {
                d_ini = 40.;
                d_fin = 115;
                
                if ( n_blocks > 1 ) {
                    d_ini = 50;
                    d_fin = 122;
                }
                
                if ( n_blocks > 3 ) {
                    d_ini = 80;
                    d_fin = 180;
                }
            }
            
            if ( rho >= 0.2 ) {
                d_ini = 30.;
                d_fin = 100;
                
                if ( n_blocks > 1 ) {
                    d_ini = 60;
                    d_fin = 155;
                }
                
                if ( n_blocks > 3 ) {
                    d_ini = 60;
                    d_fin = 165;
                }
            }
            
            if ( rho <= -0.2 ) {
                d_ini = 50.;
                d_fin = 150;
                
                if ( n_blocks > 1 ) {
                    d_ini = 10;
                    d_fin = 60;
                }
            }
            
            if ( real_network != 0 ) {
                
                if ( real_network == 1 ) {
                    d_ini = 50;
                    d_fin = 250;
                }
                
                if ( real_network == 3 ) {
                    d_ini = 10;
                    d_fin = 220;
                }
            }
            
            break;
            
        case 2: // SIS
            
            d_ini = 0.1;
            d_fin = 26;
            
            if ( real_network == 6 ) {  // Maier
                d_ini = 0.1;
                d_fin = 80;
            }
            
            break;
            
        case 3: // Ecology
            
            d_ini = 0.1;
            d_fin = 80; //95
            
            if ( real_network <= 0 && n_blocks == 2 ) { // sparse network
                d_ini = 10;
                d_fin = 250; //95
            }
            
            if ( real_network == 4 ) { // Dupont2003
                d_ini = 0.1;
                d_fin = 110;
            }
            
            if ( real_network == 5 ) { // Clements1923
                d_ini = 10;
                d_fin = 550;
            }
            
            break;
    }
    
    // Increment in d
    nd = params_bd->nd;
    dd = (d_fin - d_ini) / nd;
    
    // Range of ds used in the diagram
    
    if ( type_fFG != 2 ) { // In general, we do the forward and backward ways
        
        d_v = allocate_vector_double( 2 * nd, "d_v" );
        
        for ( i = 0; i < nd; i++ ) // "forward"
            d_v[i] = d_ini + i*dd;
        
        for ( i = 0; i < nd; i++ ) // "backward"
            d_v[nd+i] = d_v[nd-1] - i*dd;
        
        nd *= 2; // Final length of vector d_v
    }
    
    else { // In the SIS model we do the forward way only
        
        d_v = allocate_vector_double( nd, "d_v" );
        for ( i = 0; i < nd; i++ ) // "anada"
            d_v[i] = d_ini + i*dd;
    }
    
    // Vector with the parameter of the bif. diag. that we use in the plots
    alpha_v = allocate_vector_double( nd, "alpha_v" );
    
    params_bd->nd = nd;
    params_bd->d_v = d_v;
    params_bd->alpha_v = alpha_v;
    
    return;
}

// ---------------------------------------------
// Define the network's and results' directories
// ---------------------------------------------

void define_directories ( char *directory, char *directory_networks, char *net_name, struct params_interaction_matrix params ) {
    
    int status;
    int real_network = params.real_network;
    int n_blocks = params.n_blocks;
    
    if ( real_network != 0 )
        sprintf( directory, "%s/Real_networks/%s", directory, net_name );
    else
        sprintf( directory, "%s/%d_clusters", directory, n_blocks );
    status = mkdir( directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
    
    sprintf( directory_networks, "%s", directory );
    
    switch ( type_fFG ) {
        case 1: // WC
            sprintf( directory, "%s/WC", directory );
            break;
            
        case 2: // SIS
            sprintf( directory, "%s/SIS", directory );
            break;
            
        case 3: // Ecology
            sprintf( directory, "%s/Ecology", directory );
            break;
            
        default:
            sprintf( directory, "%s/Default", directory );
            break;
    }
    status = mkdir( directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
    
    printf( "Network's directory:\n%s\n\n", directory_networks );
    printf( "Simulations' directory:\n%s\n\n", directory );
    
    return;
}

// -------------------------
// Define the network's name
// -------------------------

void define_network_name ( char *net_name, struct params_interaction_matrix params_matrix ) {
    
    int real_network = params_matrix.real_network;
    int weighted = params_matrix.weighted;
    double h = params_matrix.h;
    double rho = params_matrix.rho;
    
    switch ( real_network ) {
            
        case 1:
            sprintf( net_name, "CElegans" );
            weighted = 1;
            break;
            
        case 2:
            sprintf( net_name, "Ciona" );
            break;
            
        case 3:
            sprintf( net_name, "Mouse" );
            break;
            
        case 4:
            sprintf( net_name, "Dupont2003" );
            break;
            
        case 5:
            sprintf( net_name, "Clements1923" );
            break;
            
        case 6:
            sprintf( net_name, "Maier2017" );
            break;
            
        default:
            sprintf( net_name, "h=%.2f_rho=%.2f", h, rho );
            break;
    }
    
    if ( weighted == 1 )
        sprintf( net_name, "%s_weighted", net_name );
    
    return;
}

// ------------------------
// Define the method's name
// ------------------------

void define_method_name ( char *method_name, char *correction_name, struct params_all params ) {
    
    switch ( params.reduction_method ) {
            
        case 1: // Homogeneous (naive)
            sprintf( method_name, "naive" );
            sprintf( correction_name, "" );
            break;
            
        case 2: // Degree
            sprintf( method_name, "degree" );
            sprintf( correction_name, "" );
            break;
            
        default: // Spectral
            sprintf( method_name, "eigenval" );
            
            if ( params.correction_red_syst == 0 )
                sprintf( correction_name, "" );
            else
                sprintf( correction_name, "_corr%d", params.correction_red_syst );
            break;
    }
    
    return;
}
