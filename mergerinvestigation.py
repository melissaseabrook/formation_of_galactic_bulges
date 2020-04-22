"""Test File for investigating methods for identifying mergers"""

import numpy as np
from numpy import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcol
import seaborn as sns
import pygtc
from PIL import Image
import cv2
from imutils import contours
from skimage import measure
import imutils
import photutils
from scipy import stats
from scipy.optimize import curve_fit
import scipy.ndimage as ndi
from scipy.interpolate import griddata, UnivariateSpline
import pylab
import networkx as nx
from astropy.cosmology import Planck13
import statmorph
from astropy.modeling import models, fitting

#sns.set_style('whitegrid')
def logx(x):
    if x !=0:
        if x>0:
            return np.log10(x)
        if x<0:
            return -np.log10(-x)
    else:
        return 0

def divide(x,y):
    if y !=0:
        return x/y
    else:
        return 0

def invert(var):
    if var != 0:
        return(1/var)*10
    else:
        return 0

def specificgalaxyplot(df, galaxyid, param1, param2, param3, param4):
    df2=df[df.ProjGalaxyID==galaxyid]
    df2=df2[df2.n_total>0]

    x = df2['z'].tolist()
    y_image=np.zeros(df2.z.nunique())
    y1= df2[param1].tolist()
    y2 = df2[param2].tolist()
    y3 = df2[param3].tolist()
    y4 = df2[param4].tolist()
    paths = df2['filename'].tolist()

    fig, (ax1,ax0) = plt.subplots(2,1, gridspec_kw={'height_ratios': [7.8, 1]}, sharex=True, figsize=(9,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax4=ax1.twinx()
    
    axes=[ax1,ax2,ax3,ax4]
    ax2.spines['right'].set_position(('axes', -0.25))
    ax3.spines['right'].set_position(('axes', -0.45))
    ax4.spines['right'].set_position(('axes', -0.65))
    axes[-1].set_frame_on(True)
    axes[-1].patch.set_visible(False)

    ax1.plot(x, y1,  'r', label=param1)
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylabel(param1)

    ax2.plot(x, y2,  'b--', label=param2)
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(axis='y', colors='b')
    ax2.set_ylabel(param2, labelpad=-40)

    ax3.plot(x, y3,  'g--', label=param3)
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.set_ylabel(param3, labelpad=-45)

    ax4.plot(x, y4,  'y', label=param4)
    ax4.yaxis.label.set_color('y')
    ax4.tick_params(axis='y', colors='y')
    ax4.set_ylabel(param4, labelpad=-50) 

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    for x0, y0, path in zip(x, y_image,paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax0.add_artist(ab)
    ax0.yaxis.set_visible(False) 
    ax0.set_ylim(-0.01,0.01) 
    ax0.set_xlabel('z')
    #ax.plot(x, y, ax)
    plt.subplots_adjust(left=0.4,hspace=0)
    plt.title('Pictorial Evolution of Galaxy'+str(galaxyid))
    plt.draw()
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/PictorialEvolutionGalaxy'+str(galaxyid)+'.png')
    plt.show()

def specificgalplotratesofvariabless(df, galaxyid):
    df=df[df.ProjGalaxyID==galaxyid]
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True, figsize=(8,6))
    ax0=ax1.twiny()
    ax0.set_xlabel('Lookback time (Gyr)')
    ax0.set_xticks(df.lbt)
    ax0.set_xlim(df.lbt.max(), df.lbt.min())
    ax1.plot(df.z,df.SFR, 'r' ,label='SFR')
    ax1.plot(df.z, df.dSIMdt,'purple', label='dSIMdt' )
    ax1b=ax1.twinx()
    ax1b.plot(df.z, df.dSFRdz, 'r--',label='dSFRdz')
    ax1b.set_ylabel('$\dfrac{dSFR}{dt}$')
    ax1.set_ylabel('$M_{\odot}yr^{-1}$')
    ax2.plot(df.z,df.BHAccretionrate,'brown', label='BHAccretionRate')
    ax2b=ax2.twinx()
    ax2b.plot(df.z,df.dBHmassdt, 'y', label='dBHmassdt')
    ax2b.set_ylabel('$\dfrac{dBHmass}{dt}$')
    ax2.set_ylabel('$M_{\odot}yr^{-1}$')
    ax3.plot(df.z,df.DiscToTotal, 'b', label='DiscToTotal')
    ax3.plot(df.z, df.dD2Tdz, 'b--',label='dD2Tdz')
    ax3.plot(df.z,df.n_total, 'g', label='n_total' )
    ax3b=ax3.twinx()
    ax3b.plot(df.z,df.dn_totaldz,'g--', label='dn_totaldz')
    ax3b.set_ylabel('$\dfrac{dn_total}{dt}$')
    ax3.set_xlabel('z')
    fig.legend(bbox_to_anchor=(1.01,0.5),loc='center right')
    ax0.set_title('Rate of Change of Variables for galaxy '+str(galaxyid))
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0, right=0.6)
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/RatesofVariables'+str(galaxyid)+'.png')
    plt.show()

def specificgalplotmasses(df, galaxyid):
    df=df[df.ProjGalaxyID==galaxyid]
    plt.plot(df.z,df.Starmass, label='Starmass')
    plt.plot(df.z, df.BHmass, label='BHmass')
    plt.plot(df.z, df.Gasmass, label='Gasmass')
    plt.plot(df.z, df.StellarInitialMass, label='StellarInitialMass')
    plt.xlabel('z')
    plt.yscale('log')
    plt.ylabel('$M_{\odot}$')
    plt.legend()
    plt.title('Mass Comparisons for galaxy '+str(galaxyid))
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/Massesof'+str(galaxyid)+'.png')
    plt.show()

def hierarchy_pos(G,  df, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 1):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0.8, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
        #pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
    #print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
        #pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax,  df.loc[df['DescGalaxyID'] == node]['lbt'].item())
    return pos

def plotmergertree(df, galaxyid, colourparam):
    df2=df[df.ProjGalaxyID==galaxyid]
    #df2=df2.sort_values(by=['lbt'], ascending=False)
    #df2=df2.set_index('DescGalaxyID')

    fig, ax=plt.subplots()
    G=nx.from_pandas_edgelist(df=df2, source='DescGalaxyID', target='DescID', create_using=nx.Graph)
    G.add_nodes_from(nodes_for_adding=df2.DescGalaxyID.tolist())
    #df2=df2.reindex(G.nodes())
    tree=nx.bfs_tree(G,galaxyid)
    zlist=df2.lbt.tolist()
    
    pos=hierarchy_pos(tree, df2, root=galaxyid)
    nx.draw_networkx(G, pos=pos, with_labels=False, font_size=9, node_size=50, node_color=df2[colourparam], cmap=plt.cm.plasma, vmin=df2[colourparam].min(), vmax=df2[colourparam].max(), ax=ax)
    sm=plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=df2[colourparam].min(), vmax=df2[colourparam].max()))
    sm.set_array([])
    
    ax.tick_params(left=True, labelleft=True)
    locs, labels = plt.yticks()
    print('locs={}, labels={}'.format(locs,labels))
    print(df2.z.min())
    #labels2 = np.linspace(-df2.lbt.max(), -df2.lbt.min(), len(locs))
    #labels2=np.around(labels2,decimals=1)
    labels2=np.array(locs)*(-1)
    labels2.sort()
    print(labels2)
    plt.yticks(locs, labels2)
    plt.ylabel('z')
    cbar=plt.colorbar(sm).set_label(colourparam)
    plt.title('Galaxy Merger Tree for galaxy'+str(galaxyid))
    plt.savefig('evolvinggalaxygraphbinmainbranch'+sim_name+'/MergerTreeforGalaxy'+str(galaxyid)+'.png')
    plt.show()

def findmergers(df):
    df=df[['z', 'ProjGalaxyID', 'DescID', 'DescGalaxyID', 'SHStarmass', 'BHmass', 'DMmass', 'Gasmass']]
    G=nx.from_pandas_edgelist(df=df, source='DescGalaxyID', target='DescID', create_using=nx.DiGraph)
    G.add_nodes_from(nodes_for_adding=df.DescGalaxyID.tolist())
    df2=df[['DescGalaxyID', 'SHStarmass', 'BHmass', 'DMmass', 'Gasmass']]
    df2=df2.drop_duplicates(subset='DescGalaxyID')
    node_attr=df2.set_index('DescGalaxyID').to_dict('index')
    nx.set_node_attributes(G, node_attr)
    for mass_type in ['BHmass', 'DMmass']:
        merger=[]
        for node1 in G.nodes:
            pred=list(G.predecessors(node1))
            if len(pred)>1:
                masses=[]
                for predecessors in G.predecessors(node1):
                    masses.append(G.nodes[predecessors][mass_type])
                merg=divide(sorted(masses, reverse=True)[1],max(masses))
            else:
                merg=np.nan
            merger.append(merg)
        merg_dict={n:m for n,m in zip(list(G.nodes()), merger)}
        nx.set_node_attributes(G, merg_dict, mass_type+'mergerfrac')
        print('set node'+mass_type)

    Starmerger=[]
    Gasmerger=[]
    Stargas=[]
    for node1 in G.nodes:
        pred=list(G.predecessors(node1))
        if len(pred)>1:
            Gasmasses=[]
            Starmasses=[]
            for predecessors in G.predecessors(node1):
                Starmasses.append(G.nodes[predecessors]['SHStarmass'])
                Gasmasses.append(G.nodes[predecessors]['Gasmass'])
            starmerg=divide(sorted(Starmasses, reverse=True)[1],max(Starmasses))
            gasmerg=divide(sorted(Gasmasses, reverse=True)[1],max(Gasmasses))
            stargas=divide(np.sum(Gasmasses), np.sum(Starmasses))
        else:
            starmerg=np.nan
            gasmerg=np.nan
            stargas=np.nan
        Starmerger.append(starmerg)
        Gasmerger.append(gasmerg)
        Stargas.append(stargas)

    merg_dict={n:m for n,m in zip(list(G.nodes()), Starmerger)}
    nx.set_node_attributes(G, merg_dict, 'Starmassmergerfrac')
    print('set node starmass')

    merg_dict={n:m for n,m in zip(list(G.nodes()), Gasmerger)}
    nx.set_node_attributes(G, merg_dict, 'Gasmassmergerfrac')
    print('set node gas mass')

    merg_dict={n:m for n,m in zip(list(G.nodes()), Stargas)}
    nx.set_node_attributes(G, merg_dict, 'Stargasmergerfrac')

    print('set node gas star')
    df3=pd.DataFrame.from_dict(G.nodes(), orient='index')
    df3['DescGalaxyID']=df3.index
    df3=df3[['DescGalaxyID', 'Starmassmergerfrac',  'BHmassmergerfrac',  'DMmassmergerfrac',  'Gasmassmergerfrac']]
    #print(df3)
    return df3

def plotbulgetodisc(df, sim_name):
    df['lbt']=df.apply(lambda x: -round(Planck13.lookback_time(x.z).value, 1), axis=1)
    print(df.columns.values)
    #df=df[df.Starmass>0]
    #galaxyid=116736, 132918, 436444
    df=df[(df.ProjGalaxyID == 53256)]
    df3=findmergers(df)
    df3=df3.dropna()
    print(df3)
    df['logmass']=df.apply(lambda x: logx(x.SHStarmass), axis=1)
    plotmergertree(df, 53256, 'logmass')
    #plotmergertree(df, 436444, 'Starmass')
    #plotmergertree(df, 116736, 'Starmass')
    exit()


if __name__ == "__main__":
    sim_names=['RefL0050N0752']
    #query_type=mainbranch or allbranches
    for sim_name in sim_names:
        query_type='allbranches'
        read_data=True
        if(read_data):
            print('........reading.......')
            #df=pd.read_csv('mergertest.csv')
            df=pd.read_csv('evolvingEAGLEimagesallbranchesdf'+sim_name+'.csv')
            df=df[(df.ProjGalaxyID == 116736)| (df.ProjGalaxyID == 414047)|(df.ProjGalaxyID == 436444)| (df.ProjGalaxyID ==1166619)| (df.ProjGalaxyID ==53256)]
            mergerdf= findmergers(df)
            df=pd.merge(df, mergerdf, on=['DescGalaxyID'], how='left').drop_duplicates()
            #df.to_csv('evolvingEAGLEmerger'+query_type+'df'+sim_name+'.csv')

        else:
            print('........writing.......')

        plotbulgetodisc(df, sim_name)


