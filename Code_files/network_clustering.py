# -*- encoding: utf-8 -*-
# Ara podem posar accents als comentaris!

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab
import sys
import networkx as nx
from cdlib import classes as cdlib_classes
from cdlib import algorithms
from cdlib import evaluation

# Format for figures
ftypes = ["jpg", "pdf"]

# Figure dimensions
wfig0 = 30/2.54   # conversion to inches
hfig0 = 20/2.54

# Ad-hoc colors
verd = (0.5,0.7,0)
taronja = (0.9,0.6,0)
taronja_vermellos = (231/256., 127/256., 35/256.)
vermell = (0.8,0.3,0)
vermell_lilos = (0.65,0.25,0.35)
lila = ( 0.5, 0.2, 0.7)
blau_fosc = ( 0.3, 0.35, 0.7)
blau = (0.1,0.5,0.7)
verd_fosc = (0.3, 0.5, 0.2)
verd_blau = (0.2, 0.5, 0.45)
groc = (1, 0.8, 0.2)
ocre = (0.8,0.7,0)
ocre_fosc = (217/256., 158/256., 13/256.)
negre = 'black'
gris = (0.5, 0.5, 0.5)
gris_fosc = (0.35, 0.35, 0.35)
gris_clar = (0.65, 0.65, 0.65)
blau_gris = (153/256., 196/256., 208/256.)
blau_gris_fosc = (11/256., 135/256., 170/256.)

# Tick and font sizes
mida_ticks = 35
mida_text = 45

# Font type
plt.rc("font", family='DejaVu Sans', size=mida_ticks, weight='light')

# Line widths
gruix_linia = 2

# Point sizes
mida_punt_gran = 52

plt.ion()

#......................................................................
#                               FUNCTIONS
#......................................................................

# ------------------
# Create a directory
# ------------------
def create_directory ( dir ):

    import os

    try:
        os.mkdir(dir)
    except OSError:
        print ("Creation of the directory %s failed" % dir)
        sys.exit()
    else:
        print ("Successfully created the directory %s " % dir)

    return

# ----------------------------------------
# Define N graded colors between c1 and c2
# ----------------------------------------
def define_graded_colors( c1, c2, N ):
    
    colors_N = [c1]
    
    if N > 1:
        
        dr = (c2[0]-c1[0]) / (N-1)
        dg = (c2[1]-c1[1]) / (N-1)
        db = (c2[2]-c1[2]) / (N-1)
        
        cc0 = c1[0]
        cc1 = c1[1]
        cc2 = c1[2]
        
        for j in range(N):
            
            cc0 += dr
            cc1 += dg
            cc2 += db
            
            colors_N.append( (cc0,cc1,cc2) )

    return( colors_N )

# -------------------------------------
# Define a vector of 2 alternate colors
# -------------------------------------
def define_alternate_colors( c1, c2, N ):
    
    colors_N = []
    
    for i in range( int(N/2) ):
    
        colors_N.append( (c1[0], c1[1], c1[2]) )
        colors_N.append( (c2[0], c2[1], c2[2]) )

    if N % 2 != 0: # N odd
        colors_N.append( c1 )

    return( colors_N )

# ----------------------------------------------------------------------------------
# Redefine the elements in a list so that they are consecutive numbers starting at 0
# ----------------------------------------------------------------------------------
# The list is modified and the number of different elements it contains is returned
#       Example:
#       initial list:                   [2, 1, 5, 3, 3]
#       final list:                     [1, 0, 3, 2, 2]
#       num. of different elements:     4

def redefine_indices ( list ):
    
    from collections import Counter
    
    n = len(list)
    d = { i: list[i] for i in range(n) }
    
    # Sort
    ord = sorted( d.items(), key=lambda x: x[1] )
    
    k = 0
    ord_nova = [ ( ord[0][0], 0 ) ]
    
    for i in range( 1, n ):
        
        if ( ord[i][1] != ord[i-1][1] ):
            k += 1
        ord_nova.append( (ord[i][0], k) )
    
    for i in range(n):
        pos = ord_nova[i][0]
        list[pos] = ord_nova[i][1]
    
    nelem = len( Counter(list).keys() )
    
    return( nelem )

# --------------------------
# Import a graph from a file
# --------------------------
# The first column is the node of origin and the second, the final node

def import_graph_from_file ( fname ):

    G = nx.DiGraph()
    
    # Edges
    origen, final = pylab.loadtxt( fname=fname, usecols=(0,1), unpack=True, dtype ="int" )
    llista = [ (origen[i], final[i]) for i in range( len(origen) ) ]
    
    G.add_edges_from( llista )

    return( G )

# -------------------------------
# Import a graph from a .csv file
# -------------------------------
# weighted = 1  -> the weights are imported
#         != 1  -> only the binary version is imported

def import_graph_from_csv_file ( fname, weighted=0, bipartite=0, edgelist=0 ):
    
    import csv
    
    G = nx.DiGraph()
    llista = []
    
    if edgelist != 1:
        
        if weighted != 1 and bipartite != 1:
            with open( fname, newline='') as csvfile:
            
                spamreader = csv.reader(csvfile, delimiter=',')

                i = 0
                for row in spamreader:
                    
                    for j in range( len(row) ):
                        rij = int( row[j] )
                        if rij != 0:
                            llista.append( (j, i) )
                    i += 1

            csvfile.close()

            G.add_edges_from( llista )

        elif weighted == 1 and bipartite != 1:
            with open( fname, newline='') as csvfile:
            
                spamreader = csv.reader(csvfile, delimiter=',')
            
                i = 0
                for row in spamreader:
                    
                    for j in range( len(row) ):
                        rij = float( row[j] )
                        if abs(rij) > 0.00000001:
                            llista.append( (j, i, rij) )
                    i += 1
        
            csvfile.close()

            G.add_weighted_edges_from( llista )

        elif bipartite == 1:
            
            # Count the row number (gives the number of plants or pollinators)
            # The other species category has indexes starting at nfiles
            with open( fname, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                nfiles = 0
                for row in spamreader:
                    nfiles += 1
            csvfile.close()
            
            # Interactions
            with open( fname, newline='') as csvfile:
                
                spamreader = csv.reader(csvfile, delimiter=',')
                i = 0
                for row in spamreader:
                    
                    for j in range( len(row) ):
                        rij = int( row[j] )
                        if rij != 0:
                            llista.append( (j+nfiles, i) )
                            llista.append( (i, j+nfiles) )
                    i += 1
        
            csvfile.close()
        
            G.add_edges_from( llista )

    else:
        with open( fname, newline='') as csvfile:
        
            spamreader = csv.reader(csvfile, delimiter=',')
            
            c = 0
            for row in spamreader:
                
                if c == 0: # Do not read the first line
                    c = 1
                
                else:
                    i = int(row[0])
                    j = int(row[1])
                    llista.append( (i, j) )
                    llista.append( (j, i) )
        
        csvfile.close()
    
        G.add_edges_from( llista )

    print( "Imported graph from\n", fname )
    return( G )

# -------------------------------------------
# Define the block to which each node belongs
# -------------------------------------------
# N: number of nodes
# n: number of blocks
# n_s: block sizes

def block_membership ( N, n, n_s ):

    group_i = []
    gr = 0
    nn = n_s[0]

    for i in range(N):
        
        if i >= nn:
            gr += 1
            nn += n_s[gr]
        
        group_i.append( gr )

    return( group_i )

# ------------------------------------------
# Create the (hidden) degrees between groups
# ------------------------------------------
# The hidden degrees (kIn, kOut) are used to define connection probabilities between nodes

# N: Node number
# rho: corr. coeff. between kIn and kOut within the same group
# n_s: group sizes
# p_rs: desired connection densities among groups

# We generate the degrees as uniform random variables. We use the following fact:
#
#      If (X,Y) is a Gaussian bivariate vector with mean (0,0), variance (1,1) and correlation rho,
#      then the transformation
#          V = Phi( X ),
#          W = Phi( Y ),
#          where Phi is the cumulative density function of a Gauss(0,1) variable
#          [ i.e., Phi(x) = 1/2 (1 + Erf[x/Sqrt[2]]) ]
#      is such that V, W are Unif[0,1] and corr(V,W) = rho (approx.)
#
# Therefore, if we want (V,W) be such that V,W are Unif[0,b] and corr(V,W) = rho, we do
#          V = b Phi(X),
#          W = b Phi(Y).

def create_degrees ( N, n, rho, n_s, p_rs ):

    from scipy.stats import multivariate_normal
    from scipy import special as sp
    
    # Function Phi
    def phi ( x ):
        return( 0.5 * ( 1 + sp.erf( x / np.sqrt(2) ) ) )
    
    # Random number generator
    rng = np.random.default_rng()
    
    # Group membership of nodes
    group_i = block_membership ( N, n, n_s )

    # Array with kIn, kOut
    kInOut = [ [] for i in range(N) ]
    
    # Create degrees
    for i in range(N):

        r = group_i[i]
        
        for s in range(n):
            
            # Mean degrees from/to block s
            m_in = n_s[s] * p_rs[r][s]
            m_out = n_s[s] * p_rs[s][r]
            
            # The degrees are Unif[0,b]
            b_in = 2 * m_in
            b_out = 2 * m_out
            
            if ( r != s ):
                kIn = b_in * rng.random()
                kOut = b_out * rng.random()
            
            else:
                
                x = rng.multivariate_normal( [0,0], [[1,rho],[rho,1]] )
                x1 = np.transpose(x)[0]
                x2 = np.transpose(x)[1]

                kIn = b_in * phi(x1)
                kOut = b_out * phi(x2)

            kInOut[i].append( [kIn, kOut] )

    return( kInOut )

# ----------------------------
# Crear un graf binari de novo
# ----------------------------

# This function creates a graph that is a mixture between a stochastic block model (h=0)
# and a block-configuration model (h=1)

# Fixats els paràmetres, la probabilitat de connexió j->i és una interpolació entre:
#      * una probabilitat que donaria lloc a una xarxa a blocs "clàssica" (h=0)
#      * una probabilitat que donaria una xarxa a blocs heterogènia on els graus
#        entre blocs vénen d'una distribució fixada (h=1).
#
# Si "i" pertany al grup "r" (Gr) i "j" pertany al grup "s" (Gs), llavors
#
#      p_hom(i,j) = p_hom (j->i | paràmetres) = p_rs,
#
#      p_het(i,j) = p_het (j->i | paràmetres) = kIn_i(s) * kOut_j(r) / c_rs,
#            c_rs = nr * ns * p_rs
#
# La probabilitat de connexió final és
#      p( j->i | paràmetres ) = (1-h) * p_hom(i,j) + h * p_het(i,j)
#
# Els paràmetres són:
#
#      (a) Una matriu nxn de probabilitats de connexió entre grups: (p_rs)_{r,s}
#      (b) Per a cada node "i", dos vectors de "graus" des de i cap a els altres grups:
#          kIn_i  = ( kIn_i(1),  ..., kIn_i(n)  )
#          kOut_i = ( kOut_i(1), ..., kOut_i(n) )
#
# Els vectors de graus es generen a partir d'una distribució donada, que ha de complir:
#
#      (i)  < kIn_i(s) >  = ns * p_rs
#      (ii) < kOut_i(s) > = ns * p_sr,
#      on Gr és el grup al qual pertany "i" i ns és el nombre de nodes dins el grup Gs.

# N: number of nodes
# n: number of blocks
# h: heterogeneity index
# rho: in/out-degree correlation coefficient
# n_s: block sizes
# p_rs: vector of connection densities among groups

def create_binary_graph ( N, n, h, rho, n_s, p_rs ):
    
    #print("\nCreating the binary graph...\n" )
    
    G = nx.DiGraph()
    
    # Random number generator
    rng = np.random.default_rng()
    
    # Block membership of nodes
    group_i = block_membership ( N, n, n_s )
    
    # Degrees from/to the different blocks
    kInOut = create_degrees ( N, n, rho, n_s, p_rs )
    
    for i in range(N):
    
        r = group_i[i]

        for j in range(N):
            
            s = group_i[j]
        
            # Connection j->i
        
            # kIn, kOut of nodes i,j from/to groups s,r
            kIn_i = kInOut[i][s][0]
            kOut_j = kInOut[j][r][1]

            p_hom = p_rs[r][s]
            c = n_s[r] * n_s[s] * p_rs[r][s]

            if p_rs[r][s] > 0.0001:
                p_het = kIn_i * kOut_j / c
            else:
                p_het = 0.

            # Connection prob. j->i
            pij = (1-h) * p_hom + h * p_het

            if rng.random() < pij:
                G.add_edge( j, i )

    # Aquest procés pot haver deixat nodes sense crear (si no es connecten amb ningú)
    # Reordenem les etiquetes dels nodes perquè siguin enters consecutius començant per 0
    N = G.number_of_nodes()
    mapping = { sorted(list(G.nodes()))[i] : i for i in range(N) }
    G = nx.relabel_nodes( G, mapping, copy=True )

    return( G )

# -----------------------------------------
# Dibuixar la matriu d'adjacència d'un graf
# -----------------------------------------

# sizes: llista amb les mides de les comunitats
def plot_adjacency_matrix ( file_name, G, graph_name, sizes ):
    
    import os.path
    from os import path
    
    wfig = 1. * wfig0
    hfig = 1. * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )

    # Si el graf és ponderat, prenem el mínim i el màxim valor de les arestes
    if nx.is_weighted(G) == True:
        
        weights = []
        for u in G.nodes:
            for v in G.nodes:
                if (v,u) in G.edges:
                    weights.append( G[v][u]["weight"] )
        w_max = np.max( weights )
    
    else:
        w_max = 1
    
    N = G.number_of_nodes()
    mat = nx.to_numpy_matrix( G, nodelist = sorted(G.nodes) )
    mat = np.transpose(mat)

    ad_mat = []
    for i in range(N):
        j = N-1-i
        ad_mat.append( mat.A[j] )

    fig = pylab.figure( figsize=(wfig,hfig) )
    
    x0 = 0.12
    y0 = 0.12
    mx = 0.75
    my = wfig * mx / hfig
    d = 0.1
    dc = 0.05
    my2 = my*0.022

    # Afegim una línia de colors per saber on acaben i comencen les comunitats
    n = len( sizes )
    alpha = 1 #0.7
    
    # Mides acumulades
    sizes_acum = [0]
    s = 0
    for i in range(n):
        s += sizes[i]
        sizes_acum.append( s )
    
    sx = [ [ sizes_acum[i], sizes_acum[i+1] ] for i in range(n) ]
    sy_0 = [ [0,0] for i in range(n) ]
    sy_1 = [ [1,1] for i in range(n) ]
    t1 = [ -sizes_acum[i+1] for i in range(n) ]
    t2 = [ -sizes_acum[i] for i in range(n) ]
    
    #colors = define_graded_colors( groc, blau, n )
    colors = define_alternate_colors( blau_gris, blau_gris_fosc, n )

    # Línies de colors per marcar les comunitats
    if n > 0:
        
        # Línia de colors horitzontal
        g2 = fig.add_axes([ x0, y0-dc, mx, my2 ] )
        for i in range(n):
            g2.fill_between( sx[i], sy_0[i], sy_1[i], color = colors[i], alpha=alpha )
        
        g2.set_xlim([0., N])
        g2.set_xticks( [] )
        g2.set_yticks( [] )
        
        g2.spines["top"].set_visible(False)
        g2.spines["bottom"].set_visible(False)
        g2.spines["right"].set_visible(False)
        g2.spines["left"].set_visible(False)

        # Línia de colors vertical
        g2 = fig.add_axes([ x0+mx+dc-my2, y0, my2, my ] )
        for i in range(n):
            g2.fill_between( [0,1], t1[i], t2[i], color = colors[i], alpha=alpha )

        g2.set_ylim([-N, 0.])
        g2.set_xticks( [] )
        g2.set_yticks( [] )

        g2.spines["top"].set_visible(False)
        g2.spines["bottom"].set_visible(False)
        g2.spines["right"].set_visible(False)
        g2.spines["left"].set_visible(False)

    # Dibuixem la matriu
    g1 = fig.add_axes([ x0, y0, mx, my ] )

    if nx.is_weighted(G) == True:
        mesh1 = g1.pcolormesh( ad_mat, cmap = 'Greys', vmin = 0, vmax = np.min([5,w_max]) ) #cmap = 'GnBu', vmin = 1.5, vmax = 2.
    else:
        mesh1 = g1.pcolormesh( ad_mat, cmap = 'Greys', vmin = 0, vmax = 1.7 )
    
    #mesh1.cmap.set_under( "white" )
    #mesh1.cmap.set_over( blau )
    
    g1.set_xticks( [] )
    g1.set_yticks( [] )

    # Guardar
    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + "." + ftype )
        print("%s" %(file_name + "." + ftype) )

    # Tancar
    plt.close()
    
    return

# ------------------------
# Plot degree distribution
# ------------------------

# partitions_in: List with the in-degree partition limits in [0,1] if we want to color them.
#                If not, partitions_in=[]
#
# cs:            Community or subcommunity of each node. If there are no communities, cs=[]

def plot_degree_distribution( file_name, d_in, d_out, graph_name, partitions_in, cs ):
    
    # Figure dimensions
    wfig = 1. * wfig0
    hfig = wfig
    fig = pylab.figure( figsize=(wfig, hfig) )
    
    # Axes limits
    N = len( d_in )
    if N <= 100:
        d_max = 70
        marques = [0,30,60]

    else:
        d_max = 100
        marques = [0,50,100]

    deg_max = max( max( d_in ), max( d_out ) )
    deg_min = min( min( d_in ), min( d_out ) )
    
    if ( deg_max > d_max ):
        d_max = deg_max

    xlim = [-2, d_max+2]
    ylim = xlim
    num_bins = 40 #int( 40 * (deg_max - deg_min) / 55 )

    # Plot sizes
    x0 = 0.24 # 0.1
    y0 = 0.2  # 0.1
    mx = my = 0.58 # 0.65
    d = 0.13
    sep = 0.02

    color = gris
    alpha = 1

    scatter = fig.add_axes( [ x0, y0, mx, my ] )
    hist_x = fig.add_axes( [ x0, y0+my+sep, mx, d ] )
    hist_y = fig.add_axes( [ x0+mx+sep, y0, d, my ] )
    
    scatter.tick_params(
        axis='both',            # changes apply to the x and y axis
        which='both',            # both major and minor ticks are affected
        bottom=True,            # ticks along the bottom edge are on
        top=False,                # ticks along the top edge are off
        right=False,
        left=True,
        labelbottom=True)    # labels along the bottom edge are on
        
    # No axes
    hist_x.axis('off')
    hist_y.axis('off')
    hist_x.spines["top"].set_visible(False)
    hist_x.spines["right"].set_visible(False)

    scatter.set_xticks( marques )
    scatter.set_yticks( marques )
    scatter.set_xlim( xlim )
    scatter.set_ylim( ylim )
    scatter.set_xlabel( "In-degree", size=mida_text-4 ) #fontweight="normal"
    scatter.set_ylabel( "Out-degree", size=mida_text-4 )

    hist_x.set_xlim( xlim )
    hist_y.set_ylim( ylim )

    scatter.xaxis.labelpad = 40
    scatter.yaxis.labelpad = 60

    # Separation between number and axes
    scatter.tick_params( axis = 'x', pad = 10 )
    scatter.tick_params( axis = 'y', pad = 10 )

    # Fig. margins
    fig.subplots_adjust(left=0.8, right=0.95, top=0.95, bottom=0.13)

    # Histograms
    bin_width = d_max/float(num_bins)
    bins = np.arange( -bin_width/2, d_max + bin_width, bin_width )

    label = ""

    # Degree variance and correlation
    st = np.cov( d_in, d_out )
    correlacio = np.corrcoef( d_in, d_out )[0][1]
    var_in = st[0][0]
    var_out = st[1][1]
    #print( "Degree statistics:\nm(kin) = %.3f\nm(kout) = %.3f\nStd(kin) = %.3f\nStd(kout) = %.3f\nCov(kin,kout) = %.3f\nCorrCoeff(kin,kout) = %.3f\n" %(np.mean(d_in), np.mean(d_out), np.sqrt(var_in), np.sqrt(var_out), st[0][1], correlacio) )
    #print( "<kin,kout> / <k> = %.3f\n" %( (st[0][1] + np.mean(d_in)*np.mean(d_out)) / np.mean(d_in) ) )
    
    # Scatter plot
    ncs = len( cs )
    if ncs > 0:
        
        dict = { i : cs[i] for i in range(ncs) }
        coms = set( val for val in dict.values() )
        colors = define_graded_colors( blau, ocre_fosc, len(coms) )
        
        j=0
        for cc in list( coms ):
            
            # Select the degrees of nodes that belong to community cc
            llista = [ k for k, c in dict.items() if c == cc ]
            d_in_cc = [ d_in[i] for i in llista ]
            d_out_cc = [ d_out[i] for i in llista ]
            
            scatter.scatter( d_in_cc, d_out_cc, color = colors[j], marker="o", s=mida_punt_gran, label = label + r'$\, \rho=%.1f$' %correlacio, alpha = alpha )
            j += 1

    else:
        scatter.scatter( d_in, d_out, color = color, marker="o", s=mida_punt_gran, label = label + r'$\, \rho=%.1f$' %correlacio, alpha = alpha )

    # Partitions
    m = len( partitions_in )
    if m > 0:
        colors = [vermell, verd, groc, blau]
        if (m-1) > 4:
            colors = define_graded_colors( blau, taronja_vermellos, m-1 )

        for i in range( m-1 ):
            rangx = xlim[1]-xlim[0]
            scatter.fill_between( [xlim[0] + rangx * partitions_in[i], xlim[0] + rangx * partitions_in[i+1]], xlim[0], xlim[1], color = colors[i], alpha = 0.07 )

    # Histograms
    hist_x.hist( d_in, bins = bins, color = color, density=True, histtype="step", lw=gruix_linia ) #  histtype="step"
    hist_y.hist( d_out, bins = bins, color = color, density=True, histtype="step", lw=gruix_linia, orientation='horizontal' )

    # Save
    if m > 0:
        file_name += "_%dpart" %(m-1)

    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + "." + ftype )
        print("%s" %(file_name + "." + ftype) )

    plt.close()
                        
    return

# ---------------------------------
# Imprimir les connexions d'un graf
# ---------------------------------

def print_graph ( fname, G ):

    N = G.number_of_nodes()
    m = G.number_of_edges()
    arestes = list( G.edges )

    # El graf és ponderat o no?
    arestes = list( G.edges )
    u = arestes[0][0]
    v = arestes[0][1]
    ponderat = "weight" in G[u][v]
    
    # Graf ponderat
    if ponderat == True:
        
        with open( fname, "w") as f:
            
            print( "# Number of nodes:\n# %d" %N, file=f );
            print( "# %12s %14s %14s" %("origin node", "end node", "weight"), end = "", file=f );
            
            for e in arestes:
                print( "\n%14d %14d %14.3f" %( e[0], e[1], G[e[0]][e[1]]["weight"] ), end = "", file=f )

    # Graf no ponderat
    else:
        with open( fname, "w") as f:
            
            print( "# Number of nodes:\n# %d" %N, file=f );
            print( "# %12s %14s %14s" %("origin node", "end node", "weight"), end = "", file=f );
            
            for e in arestes:
                print( "\n%14d %14d %14d" %( e[0], e[1], 1 ), end = "", file=f )

    f.close()
    #print( "\nGraf guardat com:\n%s" %fname )
    
    return

# ------------------------------------
# Imprimir les mides de les comunitats
# ------------------------------------

def print_community_sizes( file_name, G, communities ):

    n = len( communities )
    
    # Primer mirem si el graf té associat un núm. de comunitats o subcomunitats (prioritzem subcomunitats)
    tag = "res"
    if "nsubcoms" in G.graph:
        if G.graph["nsubcoms"] == n:
            tag = "subcomm"
        
        elif "ncoms" in G.graph and G.graph["ncoms"] == n:
            tag = "comm"

    elif "ncoms" in G.graph:
        if G.graph["ncoms"] == n:
            tag = "comm"

    # Si el nombre de comunitats o subcomunitats coincideix amb el de "coms",
    # imprimim la mida de les comunitats ordenant-les segons el seu nom
    if tag != "res":

        # Mides de les comunitats ordenades
        mV = [0 for i in range(n)]
        
        # Diccionari de nodes amb els noms de la seva comunitat/subcomunitat
        dict_coms = { u: G.nodes.data()[u][tag] for u in G.nodes }
        
        # Llista ordenada de valors de les comunitats
        com_values = sorted( list(set( dict_coms.values() ) ) )
        
        for i in range(n):
            c = com_values[i]
            for u in dict_coms:
                if dict_coms[u] == c:
                    mV[i] += 1

    else:
        mV = [ len( communities[i] ) for i in range(n) ]

    with open( file_name, "w") as f:
        
        print( "# Community number:\n# %d" %n, file=f );
        print( "\n# Community sizes:", file=f );
        for i in range(n):
            print( "%d" %mV[i], file=f )

    f.close()
    #print( "\nComunitats guardades com:\n%s" %file_name )

    return


# ---------------------------------------------------------------
#  Calcular la mida de les comunitats per ordre dels seus índexos
# ---------------------------------------------------------------

def community_sizes ( G, communities ):

    from collections import Counter

    n = len( communities )

    # Primer mirem si el graf té associat un núm. de comunitats o subcomunitats (prioritzem subcomunitats)
    tag = "res"
    if "nsubcoms" in G.graph:
        if G.graph["nsubcoms"] == n:
            tag = "subcomm"
        
        elif "ncoms" in G.graph and G.graph["ncoms"] == n:
            tag = "comm"

    elif "ncoms" in G.graph:
        if G.graph["ncoms"] == n:
            tag = "comm"

    # Si el nombre de comunitats o subcomunitats coincideix amb el de "coms",
    # reordenarem respectant l'ordre de l'etiqueta de les comunitats o subcomunitats
    if tag != "res":

        # Llista de noms de la comunitat/subcomunitat dels nodes
        llista_coms = [ G.nodes.data()[u][tag] for u in G.nodes ]
        
        # Comptem quants elements hi ha a cada comunitat
        dict_compt = Counter( llista_coms )
        
        # Ordenem per l'etiqueta de la comunitat
        compt_ord = sorted( dict_compt.items(), key=lambda x: x[0] )
        
        sizes = [ compt_ord[i][1] for i in range(n) ]

    else:
        sizes = [ len(c) for c in communities ]

    return( sizes )

# -----------------------------------------------------------
#  Reordenar els nodes d'un graf segons les comunitats "coms"
# -----------------------------------------------------------

def node_reordering( G, communities ):

    N = G.number_of_nodes()
    n = len( communities )
    
    # La llista de comunitats volem que estigui ordenada (en ordre ascendent) per l'índex de la comunitat
    # (en general això no està garantit perquè python altera l'ordre en el qual es mostren les comunitats)
    
    # Primer mirem si el graf té associat un núm. de comunitats o subcomunitats (prioritzem subcomunitats)
    tag = "res"
    if "nsubcoms" in G.graph:
        if G.graph["nsubcoms"] == n:
            tag = "subcomm"

        elif "ncoms" in G.graph and G.graph["ncoms"] == n:
            tag = "comm"

    elif "ncoms" in G.graph:
        if G.graph["ncoms"] == n:
            tag = "comm"

    # Si el nombre de comunitats o subcomunitats coincideix amb el de "coms",
    # reordenarem respectant l'ordre de les comunitats o subcomunitats
    if tag != "res":

        # Diccionari de nodes amb els noms de la seva comunitat/subcomunitat
        dict_coms = { u: G.nodes.data()[u][tag] for u in G.nodes }

        # Llista ordenada de valors de les comunitats
        com_values = sorted( list(set( dict_coms.values() ) ) )

        mapping = {}
        compt = 0
        for i in range(n):
            ord = {}
            c = com_values[i]
            for u in dict_coms:
                if dict_coms[u] == c:
                    ord[u] = c
            
            ord2 = sorted( ord.items(), key=lambda x: x[0] ) # ordenem per clau
            for j in ord2:
                mapping[ j[0] ] = compt
                compt += 1

    # Si no, dibuixem amb l'ordre per defecte
    else:
        
        # "Flatten" de les comunitats
        communities_flat = []
        for i in range(n):
            communities_flat = communities_flat + communities[i]

        # mapping que especifica l'etiqueta antiga i la nova (antiga:nova) del reordenament
        mapping={ communities_flat[i] : i for i in range(N) }

    # Graf reordenat
    G_reord = nx.relabel_nodes( G, mapping, copy=True )

    return ( G_reord )

# ----------------------------------------------
# Trobar els graus d'entrada i de sortida totals
# ----------------------------------------------
# Si el graf té comunitats, a més tornem una llista amb la comunitat de cada node

def degrees ( G ):

    cs = []
    
    if "ncoms" in G.graph:
        cs = [ G.nodes.data()[u]["comm"] for u in G.nodes ]
    
    if "nsubcoms" in G.graph:
        cs = [ G.nodes.data()[u]["subcomm"] for u in G.nodes ]

    d = list( G.in_degree(G.nodes) )
    d_in = [ d[i][1] for i in range(len(d)) ]
    
    d = list( G.out_degree(G.nodes) )
    d_out = [ d[i][1] for i in range(len(d)) ]

    return( d_in, d_out, cs )

# ------------------------------------------------------------------
# Trobar els graus d'entrada i de sortida respecte de cada comunitat
# ------------------------------------------------------------------
# El nombre de comunitats ["ncoms"] és un atribut del graf
# La comunitat ["comm"] és un atribut que té cada node. La numeració ha de començar en 0

def degrees_from_communities( G ):

    n = G.graph["ncoms"]
    
    # El graf és ponderat o no?
    arestes = list( G.edges )
    u = arestes[0][0]
    v = arestes[0][1]
    ponderat = "weight" in G[u][v]
    
    # Node degrees from/to the different communities
    kIn = { u: [0 for i in range(n) ] for u in G.nodes }
    kOut = { u: [0 for i in range(n) ] for u in G.nodes }

    # Graf ponderat
    if ponderat == True:
        for (u, v, wt) in G.edges.data('weight'):

            cu = G.nodes.data()[u]["comm"]
            cv = G.nodes.data()[v]["comm"]
            
            kOut[u][cv] = kOut[u][cv] + wt
            kIn[v][cu] = kIn[v][cu] + wt

    # Graf no ponderat
    else:
        for (u, v) in arestes:
            
            cu = G.nodes.data()[u]["comm"]
            cv = G.nodes.data()[v]["comm"]
            
            kOut[u][cv] = kOut[u][cv] + 1
            kIn[v][cu] = kIn[v][cu] + 1

    for u in G.nodes():

        G.nodes[u]['InDegs'] = kIn[u]
        G.nodes[u]['OutDegs'] = kOut[u]

    return

# ----------------------------------------------
# Assignar a un graf les dades de les comunitats
# ----------------------------------------------
def assign_communities( G, communities ):

    n = len( communities )
    
    # Introduïm les dades de les comunitats
    G.graph["ncoms"] = n
    
    for c in range( n ):
        for u in communities[c]:
            G.nodes[u]["comm"] = c

    return

# ------------------------------------------------
# Refinar una partició donada en funció dels graus
# ------------------------------------------------
# Es retorna la partició refinada
# var_in, var_out: variabilitat (en el grau) màxima permesa dins d'un subgrup

def partition_refinement( G, communities, var_in, var_out ):
    
    n = len( communities )

    # Introduïm les dades de les comunitats
    assign_communities( G, communities )

    # Calculem els graus d'entrada i sortida provinents de cada grup
    degrees_from_communities( G )

    dict_positions = { u: [0 for i in range(2*n) ] for u in G.nodes }
    
    # Nombre de subcomunitats per comunitat (llista)
    G.graph["nsubcoms"] = []
    
    # Per a cada comunitat, fem una subpartició segons els graus
    for c in range( n ):

        inV = np.transpose( [ G.nodes[u]["InDegs"] for u in communities[c] ] )
        outV = np.transpose ( [ G.nodes[u]["OutDegs"] for u in communities[c] ] )
        
        ntallsV = [0 for i in range(2*n)]

        # Tallem els graus cap a / des de cada comunitat un nombre de vegades suficient per assegurar que la màxima variabilitat és inferior a la cota desitjada
        for c2 in range(n):

            max_in = np.max( inV[c2] )
            min_in = np.min( inV[c2] )
            max_out = np.max( outV[c2] )
            min_out = np.min( outV[c2] )
            
            ncuts_in = int( ( max_in - min_in ) / var_in )
            ncuts_out = int( ( max_out - min_out ) / var_out )
            
            #print("\nmax_in: ", max_in, "\nmin_in: ", min_in, "\nmax_out: ", max_out, "\nmin_out: ", min_out )
            #print("\nncuts_in: ", ncuts_in, "\nncuts_out: ", ncuts_out )
            
            ntallsV[c2] = ncuts_in
            ntallsV[c2+n] = ncuts_out

            d_in = ( max_in - min_in ) / ( ncuts_in + 1 )
            d_out = ( max_out - min_out ) / ( ncuts_out + 1 )

            # punts de tall
            cuts_in = [ (i+1) * d_in for i in range(ncuts_in) ]
            cuts_out = [ (i+1) * d_out for i in range(ncuts_out) ]

            dict_in = { u: G.nodes[u]["InDegs"][c2] for u in communities[c] }
            dict_out = { u: G.nodes[u]["OutDegs"][c2] for u in communities[c] }

            # Ordenem els diccionaris segons el grau (ordre ascendent)
            dict_in = sorted( dict_in.items(), key=lambda x: x[1] )
            dict_out = sorted( dict_out.items(), key=lambda x: x[1] )
                
            # Busquem la posició de cada node segons els talls
            pos_in = 0
            pos_out = 0

            for i in range( len(communities[c]) ):
                
                u_in = dict_in[i][0]
                k_in = dict_in[i][1]
                
                u_out = dict_out[i][0]
                k_out = dict_out[i][1]
                    
                if pos_in < ncuts_in:
                    k_tall_in = cuts_in[pos_in]
                else:
                    k_tall_in = k_in + 1
        
                if ( pos_out < ncuts_out ):
                    k_tall_out = cuts_out[pos_out]
                else:
                    k_tall_out = k_out + 1

                # Mirem a quin grup està el node "u_in" segons els talls que hem definit pels kIns
                if ( k_in > k_tall_in ):
                    pos_in += 1
                if ( k_out > k_tall_out ):
                    pos_out += 1

                dict_positions[u_in][c2] = pos_in
                dict_positions[u_out][c2+n] = pos_out

        # Definim les subcomunitats dins de la comunitat c
        mV = [1]
        for i in range(2*n-2,-1,-1):
            mV.insert( 0, mV[0] * ( ntallsV[i+1] + 1 ) )

        # La subcomunitat d'un node amb posicions i_0, i_1, ..., i_(2n-1) és
        #      i_0 * mV[0] + i_1 * mV[1] + ... + i_(2n-1) * mV[2n-1]
        subcoms = [ sum ([ dict_positions[u][i] * mV[i] for i in range(2*n) ]) for u in communities[c] ]

        # Com que algunes subcomunitats poden haver quedat buides, redefinim els seus índexos
        nelems = redefine_indices ( subcoms )
        G.graph["nsubcoms"].append ( nelems )
    
        # Introduïm la subcomunitat relativa a la comunitat
        i = 0
        for u in communities[c]:
            G.nodes[u]["subcomm"] = subcoms[i]
            i += 1

    nsubcoms = G.graph["nsubcoms"]
    
    mV = [0]
    for i in range(0, n-1):
        mV.append( mV[-1] + nsubcoms[i] )
    
    # Definim les subcomunitats finals
    n_ref = sum( nsubcoms ) # núm. de grups en la partició refinada
    coms_ref = [ [] for i in range(n_ref) ]

    for u in G.nodes:
        cu = G.nodes[u]["comm"]
        cr = mV[cu] + G.nodes[u]["subcomm"]
        coms_ref[cr].append( u )
        G.nodes[u]["subcomm"] = cr # redefinim la subcomunitat com la subcomunitat absoluta

    # Introduïm el nombre total de comunitats i subcomunitats del graf
    G.graph["ncoms"] = n
    G.graph["nsubcoms"] = n_ref

    return( coms_ref )

# ------------------------------------------------
# Refinar una partició donada en funció dels graus
# ------------------------------------------------
# Es retorna la partició refinada
# cuts_in, cuts_out: punts on es tallaran els graus (tall inclòs al grup inferior)

def partition_refinement_2( G, communities, cuts_in, cuts_out ):
    
    n = len( communities )
    
    # Introduïm les dades de les comunitats
    assign_communities( G, communities )
    
    # Calculem els graus d'entrada i sortida provinents de cada grup
    degrees_from_communities( G )
    
    dict_positions = { u: [0 for i in range(2*n) ] for u in G.nodes }
    
    # Nombre de subcomunitats per comunitat (llista)
    G.graph["nsubcoms"] = []
    
    # Nombre de llindars
    ncuts_in = len( cuts_in )
    ncuts_out = len( cuts_out )
    
    # Per a cada comunitat, fem una subpartició segons els graus
    for c in range( n ):
        
        inV = np.transpose( [ G.nodes[u]["InDegs"] for u in communities[c] ] )
        outV = np.transpose ( [ G.nodes[u]["OutDegs"] for u in communities[c] ] )
        
        ntallsV = [0 for i in range(2*n)]
        
        # Tallem els graus cap a / des de cada comunitat un nombre de vegades suficient per assegurar que la màxima variabilitat és inferior a la cota desitjada
        for c2 in range(n):
            
            ntallsV[c2] = ncuts_in
            ntallsV[c2+n] = ncuts_out
            
            dict_in = { u: G.nodes[u]["InDegs"][c2] for u in communities[c] }
            dict_out = { u: G.nodes[u]["OutDegs"][c2] for u in communities[c] }
            
            # Ordenem els diccionaris segons el grau (ordre ascendent)
            dict_in = sorted( dict_in.items(), key=lambda x: x[1] )
            dict_out = sorted( dict_out.items(), key=lambda x: x[1] )
            
            # Busquem la posició de cada node segons els talls
            pos_in = 0
            pos_out = 0
            
            for i in range( len(communities[c]) ):
                
                u_in = dict_in[i][0]
                k_in = dict_in[i][1]
                
                u_out = dict_out[i][0]
                k_out = dict_out[i][1]
                
                if pos_in < ncuts_in:
                    k_tall_in = cuts_in[pos_in]
                else:
                    k_tall_in = k_in + 1
                
                if ( pos_out < ncuts_out ):
                    k_tall_out = cuts_out[pos_out]
                else:
                    k_tall_out = k_out + 1
                
                # Mirem a quin grup està el node "u_in" segons els talls que hem definit pels kIns
                if ( k_in > k_tall_in ):
                    pos_in += 1
                if ( k_out > k_tall_out ):
                    pos_out += 1
                
                dict_positions[u_in][c2] = pos_in
                dict_positions[u_out][c2+n] = pos_out
    
        # Definim les subcomunitats dins de la comunitat c
        mV = [1]
        for i in range(2*n-2,-1,-1):
            mV.insert( 0, mV[0] * ( ntallsV[i+1] + 1 ) )

        # La subcomunitat d'un node amb posicions i_0, i_1, ..., i_(2n-1) és
        #      i_0 * mV[0] + i_1 * mV[1] + ... + i_(2n-1) * mV[2n-1]
        subcoms = [ sum ([ dict_positions[u][i] * mV[i] for i in range(2*n) ]) for u in communities[c] ]
        
        # Com que algunes subcomunitats poden haver quedat buides, redefinim els seus índexos
        nelems = redefine_indices ( subcoms )
        G.graph["nsubcoms"].append ( nelems )
        
        # Introduïm la subcomunitat relativa a la comunitat
        i = 0
        for u in communities[c]:
            G.nodes[u]["subcomm"] = subcoms[i]
            i += 1

    # Retornem un objecte de tipus NodeClustering amb les dades de la partició refinada
    nsubcoms = G.graph["nsubcoms"]

    mV = [0]
    for i in range(0, n-1):
        mV.append( mV[-1] + nsubcoms[i] )

    # Definim les subcomunitats finals
    n_ref = sum( nsubcoms ) # núm. de grups en la partició refinada
    coms_ref = [ [] for i in range(n_ref) ]

    for u in G.nodes:
        cu = G.nodes[u]["comm"]
        cr = mV[cu] + G.nodes[u]["subcomm"]
        coms_ref[cr].append( u )
        G.nodes[u]["subcomm"] = cr # redefinim la subcomunitat com la subcomunitat absoluta
    
    # Introduïm el nombre total de comunitats i subcomunitats del graf
    G.graph["ncoms"] = n
    G.graph["nsubcoms"] = n_ref
    
    return( coms_ref )


# -------------------------------------------------------------------
# Llegir d'un fitxer les propietats d'una classificació en comunitats
# -------------------------------------------------------------------

def read_communities ( fname ):

    # Llegim el núm. de comunitats
    fitxer = open( fname, "r" )
    linia = fitxer.readline()
    linia = fitxer.readline()
    linia = linia.split(" ")
    n = int(linia[1])
    fitxer.close()

    # Llegim els nodes i les comunitats a les quals pertanyen
    ns, cs = pylab.loadtxt( fname=fname, usecols=(0,1), unpack=True, dtype ="int" )

    dict = { ns[i] : cs[i] for i in range(len(ns)) }
    coms = set( val for val in dict.values() ) # selecciona els ids. de les comunitats
    
    # Comprovem que el nombre de comunitats coincideix amb el que hem llegit
    if ( len(coms) != n ):
        print( "El nombre de comunitats llegit (%d) no coincideix amb el nombre de comunitats (%d). Sortim." %(n, len(coms)) )
        sys.exit()

    communities = []
    for com in coms:
        llista = [ node for node,c in dict.items() if c==com] # selecciona els nodes que estan a la comunitat com
        communities.append( llista )
    
    return( communities )

# --------------------------------------------------------------------------------
# Definir comunitats en un graf segons n (núm. de coms.) i n_s (mides de les coms)
# --------------------------------------------------------------------------------
# Les comunitats es defineixen agafant els nodes ordenats pel seu identificador

def define_communities ( G, n, n_s ):
    
    N = G.number_of_nodes()
    N_aux = np.sum( n_s )
    
    if N != N_aux:
        print ("The number of nodes in G does not coincide with community numbers in n_s.")
        sys.exit()

    nodes_ord = sorted( list( G.nodes() ) )

    communities = []
    index = 0
    for com in range(n):
        nodes_com = [ nodes_ord[i] for i in range( index, index + n_s[com] ) ]
        index += n_s[com]
        communities.append( nodes_com )

    # Introduïm el nombre total de comunitats del graf
    G.graph["ncoms"] = n

    return( communities )


#......................................................................
#                               PROGRAM
#......................................................................

# Create or read the graph
#   1   -> a new graph is created with the parameters defined later
#   2   -> a real graph is imported from file
#   def -> an artificial graph is imported from file
create_graph = 0

# Define the first partition into groups or communities
#   1   -> a selected algorithm is used to find communities
#   2   -> the communities are defined by n and n_s parameters
#   3   -> the communities are read from a file
#   def -> a single community is considered
find_communities = 2

# ---------------------------------------------------------------------
#                           Parent directory
# ---------------------------------------------------------------------

pdirectory0 = str( sys.argv[1] )

# ---------------------------------------------------------------------
#                           Network parameters
# ---------------------------------------------------------------------

# Number of blocks / clusters in the original graph
n = int( sys.argv[2] )

# Heterogeneity level, 0 <= h <= 1
h = float( sys.argv[3] )

# Correlation coefficient between hidden in/out-degrees within communities
# (only important when h > 0)
rho = float( sys.argv[4] )

# Real network
#   1 -> CElegans
#   2 -> Ciona
#   3 -> Mouse
#   4 -> Dupont2003
#   5 -> Clements1923
#   6 -> Maier2017
# def -> No real net.
real_network = int( sys.argv[5] )

if real_network in [1,2,3,4,5,6]:
    create_graph = 2
    find_communities = 3

# Weighted graph
weighted = 0

# Node number
N = 200

# n_s: community sizes
# p_s: average connection densities between communities

if n == 1:
    n_s = [N]
    p_rs = [[0.3]]

elif n == 2:
    
    n_s = [ 0.5, 0.5 ]
    
    p_rs = [ [0.3, 0.05], [0.1, 0.6] ]      # NERCCS
    #p_rs = [ [0.2, 0.], [0.5, 0.3] ]       # Ecology + SIS disconnected
    #p_rs = [ [0.2, 0.001], [0.5, 0.3] ]    # Ecology + SIS connected

elif n == 4:
    n_s = [0.2, 0.3, 0.1, 0.4]
    
    p_rs = [ [0.75, 0.05, 0.6, 0.3], [0.1, 0.2, 0.03, 0.002], [0.01, 0.5, 0.9, 0.05], [0.35, 0.03, 0.1, 0.25] ] # WC

else:
    print( "n_s and p_rs are not defined for n = %d." %n )
    sys.exit()

n_s = [ int( ns * N ) for ns in n_s ]
n_s[-1] = N - sum( n_s, - n_s[-1]) # to make sure that the sum is N

'''
print( "average in-degrees:" )
din = [0 for i in range(n)]
for i in range(n):
    for j in range(n):
        din[i] += p_rs[i][j] * n_s[j]

    print( "comm. %d: " %i, din[i] )
'''

# ---------------------------------------------------------------------
#                              Directory
# ---------------------------------------------------------------------

generic_name = "connections"
graph_name = "h=%.2f_rho=%.2f" %(h, rho)
graph_title = r"$h=%.1f, \rho=%.1f$" %(h, rho)

if create_graph == 2:
    pdirectory = pdirectory0 + "/Real_networks"
else:
    pdirectory = pdirectory0 + "/%d_clusters" %n

if os.path.isdir( pdirectory ) == False:
    create_directory ( pdirectory )
else:
    print( "Directory already exists" )

if pdirectory != "":
    pdirectory = pdirectory + "/"


# ---------------------------------------------------------------------
#                       Create or read the graph
# ---------------------------------------------------------------------

# * * * * * * * * * * * * * * * * * * * * * * *
#                 Create a graph
# * * * * * * * * * * * * * * * * * * * * * * *
if create_graph == 1:
    
    G = create_binary_graph ( N, n, h, rho, n_s, p_rs )

    # Print graph
    file_name = pdirectory + generic_name + "_" + graph_name
    print_graph ( file_name + ".txt", G )

# * * * * * * * * * * * * * * * * * * * * * * *
#            Import a real graph
# * * * * * * * * * * * * * * * * * * * * * * *
elif create_graph == 2:
    
    edgelist = 0             # 1 -> if the .csv file contains only the edge list
    bipartite = 0
    weighted = 0    #  1   -> the real graph is imported with weights
                    # != 1 -> the binary version of the graph is imported
                    
    if real_network == 1:
        graph_name = "CElegans"
        complete_name = "CElegans_connectome-Chen_PNAS_2006"
        weighted = 1
    
    elif real_network == 2:
        graph_name = "Ciona"
        complete_name = "ciona_connectome-Ryan_eLife_2016"

    elif real_network == 3:
        graph_name = "Mouse"
        complete_name = "mouse_connectome-Oh_Nature_2014"

    elif real_network == 4:
        graph_name = "Dupont2003"
        complete_name = "plants_pollinators_Dupont_2003"
        bipartite = 1
        find_communities = 2

        # Communities
        n = 2
        n_s = [11, 38]
    
    elif real_network == 5:
        graph_name = "Clements1923"
        complete_name = "plants_pollinators_Clements_1923"
        bipartite = 1
        find_communities = 2
        
        # Communities
        n = 2
        n_s = [96, 275]
    
    elif real_network == 6:
        graph_name = "Maier2017"
        complete_name = "facebook_Maier_2017"
        bipartite = 0
        edgelist = 1

    else:
        print( "To read a real network, the variable real_network should be in [1,2,3,4,5,6]." )
        sys.exit()

    graph_title = graph_name

    if weighted == 1:
        pdirectory = pdirectory + graph_name + "_weighted/"
    else:
        pdirectory = pdirectory + graph_name + "/"

    # Import graph
    #fname = pdirectory + graph_name + "/" + complete_name + ".csv"
    fname = pdirectory + complete_name + ".csv"
    G = import_graph_from_csv_file ( fname, weighted, bipartite, edgelist )


# * * * * * * * * * * * * * * * * * * * * * * *
#    Import a previously generated graph
# * * * * * * * * * * * * * * * * * * * * * * *

else:
    file_name = pdirectory + generic_name + "_" + graph_name + ".txt"
    G = import_graph_from_file ( file_name )

N = G.number_of_nodes()
print( "\nNode number: ", N )
print ("Edge number: ", G.number_of_edges(), "\n" )


# ---------------------------------------------------------------------
#                        Community detection
# ---------------------------------------------------------------------

if find_communities == 1:

    print("Searching communities using a predefined algorithm...\n")
    
    coms = algorithms.infomap( G )
    #coms = algorithms.der(G, 3, .00001, 50)
    #coms = algorithms.girvan_newman(G, level=1)

    communities = coms.communities
    method_name = coms.method_name

elif find_communities == 2:

    print("Defining %d communities according to n_s =" %n, n_s, "...\n" )
    communities = define_communities ( G, n, n_s )
    method_name = "Predefined_communities"

elif find_communities == 3:

    file_name = pdirectory + "communities_" + graph_name + "_ref0.txt"
    print("Reading the communities from file:\n%s\n" %file_name )
    communities = read_communities ( file_name )
    method_name = "Read_communities"

else:
    print("Taking the whole network as the single one initial community...\n")
    communities = define_communities ( G, 1, [N] )
    method_name = "Single_community"

assign_communities( G, communities )
sizes = community_sizes ( G, communities )

# ---------------------------------------------------------------------
#                    Print graph and community properties
# ---------------------------------------------------------------------

if weighted == 1 and bipartite == 0:
    graph_name += "_weighted"
    
file_name = pdirectory + generic_name + "_" + graph_name + ".txt"
print_graph ( file_name, G )

print ("\n***** ", method_name, " *****\n"  )
print( "Num. communities: ", len( communities ) )
print( "Community sizes: ", sizes )
#print( "Node coverage: ", coms.node_coverage )
#print( "Community overlap: ", coms.overlap, "\n" )
#print( "\nCommunities:\n", communities )

# Make sure that all the nodes have been assigned to one community
if ( sum( [ len(c) for c in communities ] ) != N ):
    print( "\nError: some nodes have not been assigned to a community.\n" )
    sys.exit()

# ---------------------------------------------------------------------
#          Plot/print graph according to the community ordering
# ---------------------------------------------------------------------

# We want to print the graph so that the node ordering is given by the communities
G_reord = node_reordering( G, communities )
    
# Plot adjacency matrix
file_name = pdirectory + "ad_matrix_" + graph_name
plot_adjacency_matrix ( file_name, G, graph_title, [N] )

file_name = pdirectory + "ad_matrix_" + graph_name + "_coms_ref0_n=%d" %len( communities )
plot_adjacency_matrix ( file_name, G_reord, graph_title, sizes )

# Plot total degrees
d_in, d_out, cs = degrees ( G )
file_name = pdirectory + "degree_distr_" + graph_name

partitions_in = []
plot_degree_distribution( file_name, d_in, d_out, graph_title, partitions_in, cs )

#partitions_in = [0, 0.5, 1.]
#plot_degree_distribution( file_name, d_in, d_out, graph_title, partitions_in, cs )

#partitions_in = [0, 1./3, 2./3, 1.]
#plot_degree_distribution( file_name, d_in, d_out, graph_title, partitions_in, cs )

# File with the reordered graph's connections
file_name = pdirectory + generic_name + "_" + graph_name + "_coms_ref0.txt"
print_graph ( file_name, G_reord )
    
# File with community sizes
file_name = pdirectory + "/community_sizes_" + graph_name + "_coms_ref0.txt"
print_community_sizes ( file_name, G, communities )


# ---------------------------------------------------------------------
#            Refining the partition according to the degrees
# ---------------------------------------------------------------------

# We do successive refinements
# In each refinement we reduce the allowed in/out-degree variability by a factor f_in/f_out

# Number of refinements
nrefins = 5

# Allowed degree variability in the first refinement
var_in = 30
var_out = 200

# Factors to reduce the allowed degree variability in successive refinements
f_in = 0.75
f_out = 1

nn0 = len( communities )
i = 0

# Alternatively, we can define a set of degree thresholds to separate nodes
# In this case, we have to use the "partition_refinement_2" function
cuts_in = [ 0 ] # [2,4]
cuts_out = []

while i < nrefins:
    
    assign_communities( G, communities )
    degrees_from_communities( G )
    communities_ref = partition_refinement( G, communities, var_in, var_out )
    #communities_ref = partition_refinement_2( G, communities, cuts_in, cuts_out )
    
    sizes_ref = community_sizes ( G, communities_ref )
    nn = len( communities_ref )
    
    var_in *= f_in
    var_out *= f_out

    if ( nn == nn0 ):
        continue
    
    print ("\n***** %s, ref. %d *****\n" %(method_name, i+1)  )
    print( "Num. communities: ", nn )
    print( "Community sizes: ", sizes_ref )

    # File with community sizes
    file_name = pdirectory + "/community_sizes_" + graph_name + "_coms_ref%d.txt" %(i+1)
    print_community_sizes ( file_name, G, communities_ref )

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    #         Plot/print graph according to the community ordering
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    # Graph reordered according to the recently defined communities
    G_reord_ref = node_reordering( G, communities_ref )

    # Plot adjacency matrix
    file_name = pdirectory + "ad_matrix_" + graph_name + "_coms_ref%d_n=%d" %(i+1, nn)
    plot_adjacency_matrix ( file_name, G_reord_ref, graph_title, sizes_ref )

    # File with the reordered graph's connections
    file_name = pdirectory + generic_name + "_" + graph_name + "_coms_ref%d.txt" %(i+1)
    print_graph ( file_name, G_reord_ref )

    communities = communities_ref
    i += 1
    nn0 = nn

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    #        If the refinement is close to maximal (nn > 0.98 N), quit
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    if nn > 0.98*N:
        break

# -----------------------------------------------------------------
#  Print a last partition in which each community is a single node
# -----------------------------------------------------------------

if nrefins != 0:

    # Communities
    communities = [ [u] for u in G.nodes ]
    sizes = [1 for u in G.nodes ]
    nn = len( communities )

    # File with community sizes
    file_name = pdirectory + "/community_sizes_" + graph_name + "_coms_ref%d.txt" %(i+1)
    print_community_sizes ( file_name, G, communities )

    # Plot adjacency matrix
    file_name = pdirectory + "ad_matrix_" + graph_name + "_coms_ref%d_n=%d" %(i+1, nn)
    plot_adjacency_matrix ( file_name, G, graph_title, sizes )

    # File with the reordered graph's connections
    file_name = pdirectory + generic_name + "_" + graph_name + "_coms_ref%d.txt" %(i+1)
    print_graph ( file_name, G )
