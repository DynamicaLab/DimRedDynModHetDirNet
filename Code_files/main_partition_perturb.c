# include "header.h"

// Marina Vegué Llorente
// April 8 2020

//----------------------------------------------------------------------------------//
//                           SYSTEM OF ODEs on a graph                              //
//----------------------------------------------------------------------------------//
//                                                                                  //
//    Fundamentals:                                                                 //
//                                                                                  //
//    * The network has N nodes.                                                    //
//                                                                                  //
//    * Each node's activity obeys:                                                 //
//                                                                                  //
//       tau * dxi / dt = F(xi) + Sum_j [ d_{ij} * G(xi, xj) ],                     //
//                                                                                  //
//       where ( d_{ij} )_{ij} is the interaction matrix.                           //
//                                                                                  //
//    * Nodes are assumed to be partitioned into groups. Both the interaction       //
//      matrix and the groups' sizes are read from files assuming that the node's   //
//      indices are ordered according to the group membership.                      //
//                                                                                  //
//----------------------------------------------------------------------------------//

//int main( ) {

int main( int argc, char *argv[] ) {
        
    int i, j, i_f, N, n, n_blocks, steps, *sizes, sim_time[3], reduction_method, read_coms, real_network, error_analysis, ref_index, np, nf, pert, correction_red_syst, weighted=0;
    double *yMacro;
    double h, rho, t_sim, tau, start, end;
    double x_ini0, x_ini1, d0, f, df, error, average;
    char subdirectory[300], directory[300], directory_networks[300], name[300], name_red[300], net_name[50], coms_name[100], method_name[100], name_errors[300], correction_name[100];
    gsl_matrix *mMat, *dMat_or, *dMat_or0;
    gsl_vector_int *group_indices;
    
    struct params_all params;
    struct params_cond_ini params_ini;
    struct params_interaction_matrix params_matrix;
    struct params_bif_diag params_bdiag;
    
    FILE *fout;
    
    start = clock();
    srand( (int) time (NULL) );
    INITIALIZE_RANDOM;
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                          SIMULATION PARAMETERS                              //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    t_sim = 30; //50; // Simulation time
    steps = 2;  // Number of steps of the trajectory that will be printed
    
    error_analysis = 0; // 1 -> compare the error in the comp. eqs. to the errors resulting from taking random vectors
    
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                         INTERACTION MATRIX PARAMETERS                       //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    n_blocks = atoi((char*)argv[2]);         // Number of blocks in the original network
    //n_blocks = 2;
    
    // Weight of non-exsisting connections (to make the interaction matrix positive)
    d0 = 0.0001;
    
    // Network type
    
    // Artificial networks
    // Heterogenity level of the imported matrix. It interpolates between the cases
    //    * h=0 -> SBM
    //    * h=1 -> Block version of the Chung Lu model
    //h = 0.;
    h = atof((char*)argv[3]);
    
    // In/out-degree correlation coefficient
    //rho = 0.;
    rho = atof((char*)argv[4]);
    
    // Real networks
    //  * 1 -> CElegans
    //  * 2 -> Ciona
    //  * 3 -> Mouse
    //  * 4 -> Dupont2003
    //  * 5 -> Clements1923
    //  * 6 -> Maier2017
    //  * def -> No real net.
    //real_network = 0;
    real_network = atoi((char*)argv[5]);
    
    params_matrix.h = h;
    params_matrix.rho = rho;
    params_matrix.real_network = real_network;
    params_matrix.n_blocks = n_blocks;
    params_matrix.weighted = weighted;
    
    define_network_name ( net_name, params_matrix );

    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                      DIMENSION REDUCTION PARAMETERS                         //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    //read_coms = 1;
    read_coms = atof((char*)argv[6]);   //   1  -> Read a network organized into groups (communities)
                                        //  def -> Read a network without groups and do the 1-dim reduction
    //ref_index = 0;
    ref_index = atoi((char*)argv[7]); // Refinement index when read_coms = 1
    
    // Partition name
    switch ( read_coms ) {
            
        case 1:
            sprintf( coms_name, "_coms_ref%d", ref_index );
            break;
            
        default:
            sprintf( coms_name, "" );
            break;
    }
    
    // Reduction method
    //      1: Homogeneous
    //      2: Degree-based
    //      def: Spectral
    //reduction_method = 0;
    reduction_method = atoi((char*)argv[8]);
    
    // We also include the corrections of reduced system for the spectral method:
    //      1 -> correction in which the 1st-order terms are cancelled and the Taylor approx. points
    //           are modified (like in Laurence et al., Thibeault et al.)
    //      2 -> correction in which the Taylor points are the observables but we consider
    //           the 1st-order terms
    //    def -> no correction: the reduced system has the same form as the original system
    //correction_red_syst = 0;
    correction_red_syst = atoi((char*)argv[9]);
    
    params.reduction_method = reduction_method;
    params.correction_red_syst = correction_red_syst;
    
    define_method_name ( method_name, correction_name, params );
    
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                           DYNAMICS' PARAMETERS                              //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    tau = 1;
    
    // Type of F, G functions:
    //      1: Neuronal dynamics - Wilson-Cowan (WC)
    //      2: SIS
    //      3: Ecology
    //      default: F(x) = 1-x, G(x,y) = y
    type_fFG = 1;
    
    // Neuronal dynamics' parameters
    tau_fG = 0.3;
    mu_fG = 10;
    
    // SIS dynamics' parameters
    gamma_fG = 1;
    
    // Ecological dynamics' parameters
    B_fG = 0.1;
    C_fG = 1;
    K_fG = 5;
    D_fG = 6;
    E_fG = 0.9;
    H_fG = 0.1;
    
    switch ( real_network ) {
            
        case 1:
            type_fFG = 1;
            break;
            
        case 2:
            type_fFG = 1;
            break;
            
        case 3:
            type_fFG = 1;
            break;
            
        case 4:
            type_fFG = 3;
            break;
            
        case 5:
            type_fFG = 3;
            break;
            
        case 6:
            type_fFG = 2;
            break;
    }
    
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                     BIFURCATION DIAGRAM'S PARAMETERS                        //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    // The bif. diag. is computed by varying the overall strength of the connections.
    // At each step, the connections are multiplied by a constant factor d / N.
    
    params_bdiag.nd = 100; //130; // number of steps in the bif. diag. ("one way" only)
    define_bif_diag_parameters ( &params_matrix, &params_bdiag );
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //         PARAMETERS OF THE INITIAL CONDITION FOR THE ORIGINAL SYSTEM         //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    // The original condition for the activities is Unif[ x_ini0, x_ini1 ]
    x_ini0 = 15;
    x_ini1 = 20;
    
    if ( type_fFG == 3 ) {
        x_ini0 = 0;
        x_ini1 = 0;
    }
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                            WORKING DIRECTORIES                              //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    //sprintf( directory, "/Users/marina/Documents/Postdoc/OneDrive - Université Laval/Dimension_reduction" );
    //sprintf( directory, "/Users/marina/Desktop/Prova" );
    sprintf( directory, "%s", argv[1] );
    
    printf("\nRead directory:\n%s\n\n", directory );
    define_directories ( directory, directory_networks, net_name, params_matrix );

    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //           READ:        the size N,                                          //
    //                        the matrix dMat = (d_ij)_i,j                         //
    //                        the size n and the communities                       //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    // Read the original dMat matrix
    sprintf( name, "connections_%s%s.txt", net_name, coms_name );
    read_and_create_dMat_or ( &N, &dMat_or0, name, directory_networks );
    
    if ( read_coms == 1 ) {
        sprintf( name, "community_sizes_%s%s.txt", net_name, coms_name );
        read_communities ( &n, &sizes, &group_indices, name, directory_networks );
    }
    
    else {
        n = 1;
        sizes = allocate_vector_int( n, "sizes" );
        group_indices = gsl_vector_int_alloc( n ); // micro index where each group starts
        sizes[0] = N;
        gsl_vector_int_set_zero( group_indices );
    }
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                               OUTPUT FILES                                  //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    sprintf( name, "bif_diagram_%s.txt", method_name );
    sprintf( name_red, "bif_diagram_reduced_%s.txt", method_name );
    
    sprintf( name_errors, "RMSE_perturb_%s_%s_n=%d%s.txt", method_name, net_name, n, correction_name );
    fout = open_file( directory, name_errors, "w" );
    fprintf( fout, "# %24s %12s %12s\n", "frac. of nodes perturbed", "mean error", "std of error" );
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                         INITIALIZE PARAMETERS                               //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    dMat_or = gsl_matrix_alloc( N, N );
    mMat = gsl_matrix_calloc( n, N );
    
    params.dMat_or = dMat_or;
    params.mMat = mMat;
    params.N = N;
    params.n = n;
    params.tau = tau;
    params.d0 = d0;
    
    params.group_indices = group_indices;
    params.ns = sizes;
    params.t_sim = t_sim;
    params.steps = steps;
    params.directory = directory;
    params.subdirectory = subdirectory;
    
    params_ini.x_ini0 = x_ini0;
    params_ini.x_ini1 = x_ini1;
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    //                                                                                   //
    //   PERTURB THE INITIAL PARTITION AND COMPUTE THE RESULTING DYNAMICAL ERROR         //
    //                                                                                   //
    //   f * N: number of node pairs whose membership is interchanged                    //
    //   np:    number of partition perturbations generated for each f                   //
    //                                                                                   //
    //   For each f we:                                                                  //
    //                                                                                   //
    //   - perturb the original partition by flipping the membership of f*N node pairs   //
    //   - compute the bifurcation diagram                                               //
    //   - compute the error in the bif. diag. (dynamical error, RMSE)                   //
    //   - write the result in a file                                                    //
    //                                                                                   //
    ///////////////////////////////////////////////////////////////////////////////////////
    
    int *map, *groups = allocate_vector_int( N, "groups" );
    gsl_vector *data_average, *data_red;
    gsl_matrix *data, *data_or;
    double *errors, mean, var;
    
    data = gsl_matrix_alloc( params_bdiag.nd, n );
    data_average = gsl_vector_alloc( params_bdiag.nd );
    data_red = gsl_vector_alloc( params_bdiag.nd );
    data_or = gsl_matrix_alloc( params_bdiag.nd, N );
    yMacro = allocate_vector_double( n, "yMacro" );
    
    nf = 10; //15;            // number of different fs
    df = 1./(nf-1);     // f increment
    np = 100; //350;           // number of indep. perturbations for each f
    
    errors = allocate_vector_double( np, "errors" );
    
    // o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
    //            DYNAMICAL ERROR FOR THE ORIGINAL (NON PERTURBED) PARTITION
    // o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
    
    // We compute the complete bif. diag. (integrating both the micro and macro systems) for the original partition.
    // For the perturbed partitions, we only integrate the macro system and we compute the exact bif. diag. by reading
    // the data computed when the partition was the original one.

    // dMat_or = dMat_or0
    gsl_matrix_memcpy( dMat_or, dMat_or0 );
    
    // Bifurcation diagram
    bifurcation_diagram ( &params, &params_ini, params_bdiag, name, name_red, 0, 0, 1 );
    
    // Read the bif. diag. data
    read_data_bif_diagram ( data_average, data_red, name, name_red, directory );
    
    // Error (root-mean-square error, RMSE) between the exact curve and that of the reduced dynamics
    error = RMSE ( data_average, data_red );
    fprintf( fout, "%26.3f %12.6f %12.6f\n", 0., error, 0. );
    
    // Read the complete data of the microscopic equilibrium according to the original partition
    read_complete_data_bif_diag ( data_or, "equilibrium_micro.txt", directory );

    
    // o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
    //                DYNAMICAL ERROR FOR THE PERTURBED PARTITIONS
    // o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o
    
    // Node permutation corresponding to the perturbed partition
    map = allocate_vector_int( N, "map" );
    
    f = df;
    for ( i_f = 1; i_f < nf; i_f ++ ) {
        
        for ( pert = 0; pert < np; pert++ ) {
        
            printf( "* * * * * * * * * * * * * * * *\n    f = %.3f, pert = %d \n* * * * * * * * * * * * * * * *\n\n", f, pert );
            
            // dMat_or = dMat_or0
            gsl_matrix_memcpy( dMat_or, dMat_or0 );
            
            // Perturb the original node partition and reorder the interaction matrix accordingly
            perturb_node_partition ( N, groups, group_indices, f );
            reorder_dMat ( dMat_or, groups, map );
            //params.dMat_or = dMat_or;
            
            // Compute the reduced bif. diagram
            bifurcation_diagram ( &params, &params_ini, params_bdiag, name, name_red, 0, 0, 0 );

            // Read the bif. diag. data (only data corresponding to the reduced system)
            read_data_bif_diagram ( NULL, data_red, name, name_red, directory );
            
            // Compute the exact bif. diagram, taking into account the node perturbation
            compute_data_bif_diag ( data_or, data, map, &params );

            // Compute the average observable for each parameter in the diagram
            for ( i = 0; i < params_bdiag.nd; i++ ) {
                
                for ( j = 0; j < n; j++ )
                    yMacro[j] = gsl_matrix_get( data, i, j );
                
                average = observable_average ( n, yMacro, sizes );
                gsl_vector_set( data_average, i, average );
            }
            
            // Compute the dynamical error, defined as the "root-mean-square error" (RMSE),
            // between the exact diagram and the reduced one
            error = RMSE ( data_average, data_red );
            errors[pert] = error;
        }
        
        mean = gsl_stats_mean( errors, 1, np );
        var = gsl_stats_variance( errors, 1, np );
        fprintf( fout, "%26.3f %12.6f %12.6f\n", f, mean, sqrt(var) );
        
        f += df;
    }
    
    // Print the basic parameters
    print_params ( params );
    
    // Free memory
    gsl_matrix_free( dMat_or );
    gsl_matrix_free( dMat_or0 );
    gsl_matrix_free( mMat );

    free( params_bdiag.d_v );
    free( params_bdiag.alpha_v );
    
    free( yMacro );
    free( group_indices );
    free( groups );
    
    free( map );
    free( errors );
    gsl_vector_free( data_average );
    gsl_vector_free( data_red );
    gsl_matrix_free( data );
    gsl_matrix_free( data_or );
    
    fclose( fout );
    
    FREE_RANDOM;
    
    // Execution time
    end = clock();
    convert_time( (end-start)/CLOCKS_PER_SEC, sim_time );
    printf( "\nExecution time: %d h, %d min, %d sec\n\n", sim_time[0], sim_time[1], sim_time[2] );
    
    return(0);
}
