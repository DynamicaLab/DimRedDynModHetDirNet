
------------------
0. Before starting
------------------

* pdirectory is the parent directory where all the files will be stored

* To work on real networks, inside pdirectory there has to be a folder called "Real_networks".
  Inside this folder, the following folders and files should exist:
  
  CElegans_weighted/CElegans_connectome-Chen_PNAS_2006.csv
  CElegans_weighted/communities_CElegans_ref0.txt

  Ciona/ciona_connectome-Ryan_eLife_2016.csv
  Ciona/communities_Ciona_ref0.txt

  Mouse/mouse_connectome-Oh_Nature_2014.csv
  Mouse/communities_Mouse_ref0.txt

  Clements1923/plants_pollinators_Clements_1923.csv

  Dupont2003/plants_pollinators_Dupont_2003.csv

  Maier2017/facebook_Maier_2017.csv
  Maier2017/communities_Maier2017_ref0.txt

* To compare the solutions to the comp. eqs. when it is restricted to the subspace spanned by
  the leading eigenvector vs when it is not (for randomly generated matrices), there has to be
  a folder called "Tests". Inside this folder, the following files should exist:

  sols_compEq_distances_m=m0_r=r0.txt
  sols_compEq_errors_m=m0_r=r0.txt
	
  for (m0, r0) in [ (2,2), (4,2), (6,2), (3,3), (5,3), (7,3), (5,5), (7,5), (9,5) ]

* Parameters used as arguments in all the scripts:

  pdirectory	   parent directory

  real_network	   index of the real network:
		   1   -> CElegans
		   2   -> Ciona
		   3   -> Mouse
		   4   -> Dupont2003
		   5   -> Clements1923
		   6   -> Maier2017
		   def -> No real network

  n_blocks	   number of blocks
  h		   heterogeneity level
  rho		   corr. coeff. between (hidden) in- and out-degrees 
		   (important for artificial networks only)

  read_coms	   use a node partition or not
		   1   -> read a partition from file
		   def -> consider the whole network as a single community

  ref_index	   index of the refinement considered
		   (only important when read_coms = 1)

  reduction_meth   method used for dimension reduction
		   1   -> homogeneous reduction
	     	   2   -> degree-based reduction
	     	   def -> spectral reduction
	     
  correction	   correction used when the method is the spectral one
		   1   -> correction that changes Taylor points and removes 1st-order terms
		   2   -> correction that preserves Taylor points and includes 1st-order terms
		   def -> no correction
		   Note: in the paper we use correction 2; if the dynamics is WC, corrs. 2 and def coincide


-----------------------------------------------------
1. Read or create the graph and successive partitions
-----------------------------------------------------

Use the python script "network_clustering.py"

The arguments to the python script are:
   pdirectory | n_blocks | h | rho | real_network

To work on a real network (CElegans, Ciona, Mouse, Dupont2003, Clements1923 or Maier2017),
"real_network" has to be in [1,2,3,4,5,6].

To work on an artificial network, real_network has to be 0.
The parameters n_blocks, h, rho define the properties of this network, either if it is read
from file (create_graph=1) or newly created (create_graph=0).

From a given initial partition, successive refinements are created.
The initial partition can be either:
    * computed using a selected algorithm (find_communities = 1)
    * defined using a predefined set of groups (find_communities = 2)
    * read from file (find_communities = 3)
    (otherwise a single community is taken as the original partition)

For every partition:
    * nodes are reordered so that nodes in the same group have consecutive indices
    * the adjacency matrix is plotted according to this reordering
    * the properties of the reordered graph and the partition are printed in files
      that will be used later to compute the bif. diagrams


----------------------------------
2. Compute the bifurcation diagram
----------------------------------

Files to be compiled:
main.c, functions.c, header.h, random.h

The executable file computes the bif. diag. for a given network and node partition, for all
the methods (spectral, homogeneous and degree if n=1) and for all the corrections (0, 1 and 2).

To compute the error associated to the comp. eqs. (for the first parameter in the bif. diag. only),
the variable "error_analysis" has to be set to 1.

The arguments to the executable file are:
   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index


---------------------------------------------------------------------
3. Compute the errors in the bif. diag. as the partition is perturbed
---------------------------------------------------------------------

Files to be compiled:
main_partition_perturb.c, functions.c, header.h, random.h

The executable file computes the bif. diag. for a succession of partitions that are
obtained by perturbing an original partition.

The reduction method and the correction used in the spectral method are given as arguments.

The arguments to the executable file are:
   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index |
   reduction_meth | correction

 
-------------------
4. Plot the results
-------------------

Use the python script "make_plots.py"

The arguments to the script are:
   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index |
   correction | what_to_plot

what_to_plot	0 -> plot the bif. diagram for a given partition and all the reductions
		1 -> plot the errors in the bif. diag. as a partition is perturbed
		2 -> plot the errors in the bif. diag. as the partition size is varied
		3 -> plot the errors in the compatibility equations
		4 -> compare the errors when using 2 methods for solving the comp. eqs. for
		     a set of randomly generated matrices (the data file must be provided)



