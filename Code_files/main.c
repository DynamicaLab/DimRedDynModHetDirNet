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
        
    int i, N, n, n_blocks, steps, *sizes, sim_time[3], read_coms, real_network, error_analysis, ref_index, print_parameters, integrate_micro, red, *map, weighted=0;
    double h, rho, t_sim, tau, start, end;
    double x_ini0, x_ini1, d0;
    char directory[300], directory_networks[300], subdirectory[300], name[300], name_red[300], net_name[50], coms_name[100], method_name[100], correction_name[100], source[300], dest[300];
    
    gsl_matrix *dMat_or;
    gsl_matrix *mMat, *data_or, *data;
    gsl_vector *data_average;
    gsl_vector_int *group_indices;
    
    struct params_all params;
    struct params_cond_ini params_ini;
    struct params_interaction_matrix params_matrix;
    struct params_bif_diag params_bdiag;
    
    start = clock();
    srand( (int) time (NULL) );
    INITIALIZE_RANDOM;
    
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //                          SIMULATION PARAMETERS                              //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    t_sim = 30; //50 // Simulation time
    steps = 2;  // Number of steps of the trajectory that will be printed
    
    error_analysis = 1; // 1 -> compare the error in the comp. eqs. to the errors resulting from taking random vectors
    
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
    //h = 1.;
    h = atof((char*)argv[3]);
    
    // In/out-degree correlation coefficient
    //rho = 0.8;
    rho = atof((char*)argv[4]);
    
    // Real networks
    //  * 1 -> CElegans
    //  * 2 -> Ciona
    //  * 3 -> Mouse
    //  * 4 -> Dupont2003
    //  * 5 -> Clements1923
    //  * 6 -> Maier2017
    //  * def -> No real net.
    //real_network = 6;
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
    
    //read_coms = 0;
    read_coms = atof((char*)argv[6]);   //   1  -> Read a network organized into groups (communities)
                                        //  def -> Read a network without groups and do the 1-dim reduction
    //ref_index = 0;
    ref_index = atoi((char*)argv[7]);   // Refinement index when read_coms = 1
    
    // Partition name
    switch ( read_coms ) {
            
        case 1:
            sprintf( coms_name, "_coms_ref%d", ref_index );
            break;
            
        default:
            sprintf( coms_name, "" );
            break;
    }
    
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
    
    // Define the type of dynamics when the network is real
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
    
    params_bdiag.nd = 120; // number of steps in the bif. diag. ("one way" only)
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
    
    //sprintf( directory, "/Users/marina/Documents/Postdoc/OneDrive - Université Laval/Dimension_reduction" ); // main directory
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
    read_and_create_dMat_or ( &N, &dMat_or, name, directory_networks );
    
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
    //                         INITIALIZE PARAMETERS                               //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    mMat = gsl_matrix_calloc( n, N );
    
    params.mMat = mMat;
    params.dMat_or = dMat_or;
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
    
    
    /////////////////////////////////////////////////////////////////////////////////
    //                                                                             //
    //           BIFURCATION DIAGRAM FOR THE DIFFERENT REDUCTION METHODS           //
    //                                                                             //
    /////////////////////////////////////////////////////////////////////////////////
    
    data_or = gsl_matrix_alloc( params_bdiag.nd, N );
    data = gsl_matrix_alloc( params_bdiag.nd, n );
    data_average = gsl_vector_alloc( params_bdiag.nd );
    map = allocate_vector_int( N, "map" );
    
    for ( i = 0; i < N; i++ )
        map[i] = i;
    
    // In general, we compute the diagrams for the methods SPECTRAL (eigenval) and HOMOGENEOUS (naive)
    // If n = 1, we do it for the degree-based method (Gao. et al.) as well
    // We only integrate the complete system in the first case
    int method_number = 2;
    if ( n == 1 )
        method_number = 3;
    
    // We also include the corrections of reduced system for the spectral method:
    //      1 -> correction in which the 1st-order terms are cancelled and the Taylor approx. points
    //           are modified (like in Laurence et al., Thibeault et al.)
    //      2 -> correction in which the Taylor points are the observables but we consider
    //           the 1st-order terms
    //    def -> no correction: the reduced system has the same form as the original system
    method_number += 2;
    
    for ( red = 0; red < method_number; red++ ) {
        
        switch (red) {
           
            case 1: // Spectral + correction 1
                params.reduction_method = 0;
                params.correction_red_syst = 1;
                integrate_micro = 0;
                break;
                
            case 2: // Spectral + correction 2
                params.reduction_method = 0;
                params.correction_red_syst = 2;
                integrate_micro = 0;
                break;
                
            case 3: // Homogeneous (naive)
                params.reduction_method = 1;
                params.correction_red_syst = 0;
                integrate_micro = 0;
                break;
                
            case 4: // Degree
                params.reduction_method = 2;
                params.correction_red_syst = 0;
                integrate_micro = 0;
                break;
                
            default: // Spectral
                params.reduction_method = 0;
                params.correction_red_syst = 0;
                integrate_micro = 1;
                break;
        }
        
        define_method_name ( method_name, correction_name, params );
    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //                          OUTPUT FILES
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        sprintf( name, "bif_diagram_%s_%s_n=%d%s.txt", method_name, net_name, n, coms_name );
        sprintf( name_red, "bif_diagram_reduced_%s_%s_n=%d%s%s.txt", method_name, net_name, n, coms_name, correction_name );
        
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //                        BIFURCATION DIAGRAM
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        print_parameters = 1;
        bifurcation_diagram ( &params, &params_ini, params_bdiag, name, name_red, error_analysis, print_parameters, integrate_micro );
        
        // If the method is the spectral wihtout correction, we read the complete exact equilibrium
        if ( params.reduction_method != 1 && params.reduction_method != 2 && params.correction_red_syst != 1 && params.correction_red_syst != 2 )
            read_complete_data_bif_diag ( data_or, "equilibrium_micro.txt", directory );

        // For the homogeneous and degree methods, we compute the data of the bif. diagram corresponding to the
        // complete system from the previous data
        if ( params.reduction_method == 1 || params.reduction_method == 2 ) {
            compute_data_bif_diag ( data_or, data, map, &params );
            
            int j, average_only;
            FILE *f_output;
            double *yMacro = allocate_vector_double( n, "yMacro" );
            
            f_output = open_file( directory, name, "w" );
            fprintf( f_output, "# %10s %17s %30s\n", "Param.", "Average node obs.", "Obs. from the micro. system" );
            
            average_only = 0;
            for ( i = 0; i < params_bdiag.nd; i++ ) {
    
                for ( j = 0; j < n; j++ )
                    yMacro[j] = gsl_matrix_get( data, i, j );
                
                print_equilibrium( f_output, params_bdiag.alpha_v[i], n, n, yMacro, sizes, average_only );
            }
            
            fclose( f_output );
            free( yMacro );
        }
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //                        ERROR ANALYSIS
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        // Copy outside of the folder the file with the errors in the comp. eqs. of
        // the spectral method for the first d
        if ( error_analysis == 1 && params.reduction_method != 1 && params.reduction_method != 2 && params.correction_red_syst != 1 && params.correction_red_syst != 2 ) {
            
            sprintf( source, "%s/errors_random_observables_n=%d.txt", params.subdirectory, n );
            sprintf( dest, "%s/errors_random_observables_n=%d.txt", directory, n );
            copy_file ( source, dest );
        }
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //              PRINT THE REDUCED SYSTEM'S PARAMETERS
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        if ( print_parameters == 1 ) {
            
            // Copy the observable matrix when the method is not the homogeneous one
            if ( params.reduction_method != 1 ) {
                sprintf( source, "%s/observable_vectors.txt", params.subdirectory );
                sprintf( dest, "%s/observable_vectors_%s_%s_n=%d.txt", directory, method_name, net_name, n );
                copy_file ( source, dest );
            }
            
            // Copy the reduced matrix (delta matrix)
            sprintf( source, "%s/delta_matrix.txt", subdirectory );
            sprintf( dest, "%s/delta_matrix_%s_%s_n=%d.txt", directory, method_name, net_name, n );
            copy_file ( source, dest );
        }
    }
    
    // Print the basic parameters
    print_params ( params );
    
    // Free memory
    gsl_matrix_free( dMat_or );
    gsl_matrix_free( mMat );
    gsl_matrix_free( data_or );
    gsl_vector_free( data_average );
    gsl_matrix_free( data );
    
    free( params_bdiag.d_v );
    free( params_bdiag.alpha_v );
    free( group_indices );
        
    FREE_RANDOM;
    
    // Execution time
    end = clock();
    convert_time( (end-start)/CLOCKS_PER_SEC, sim_time );
    printf( "\nExecution time: %d h, %d min, %d sec\n\n", sim_time[0], sim_time[1], sim_time[2] );
    
	return(0);
}
