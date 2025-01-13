from pathlib import Path
import networkx as nx
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import pickle as pk
import numpy as np
import pandas as pd
from datetime import datetime as dt
from geopy.distance import great_circle
import matplotlib.pyplot as plt
from random import sample
from random import choices
import seaborn as sns
import bisect
import math
import copy

def diadic_density(dem, G_input, reciprocal=False):
    G = G_input.copy()
    attr = nx.get_node_attributes(G,dem)
    nodes_rem = [key for key,val in attr.items() if pd.isna(val)]
    G.remove_nodes_from(nodes_rem)
    attr = nx.get_node_attributes(G,dem)
    cnt = Counter(list(attr.values()))
    del cnt["Unknown"]
    
    ### Calculates total number of possible dyads for each pair of attributes
    cnt_pairs = {}
    for key,val in cnt.items():
        for key2, val2 in cnt.items():
            if reciprocal:
                #since it is reciprocal network, there is no difference between key2,key1 or key1,key2
                #because there is no direction
                if not (key2,key) in cnt_pairs.keys():
                    cnt_pairs[(key, key2)] = val*val2 if key!=key2 else (val*(val2-1))/2
            else:
                cnt_pairs[(key, key2)] = val*val2 if key!= key2 else val*(val2-1)
    
    ### Counts how many dyads in the network are between nodes with each possible attribute value
    cnt_ties = {key:0 for key in cnt_pairs.keys()}
    for edge in tqdm(G.edges()):
        att1 = G.nodes[edge[0]][dem]
        att2 = G.nodes[edge[1]][dem]
        if att1 != "Unknown" and att2 != "Unknown":
            if reciprocal:
                if (att1, att2) in cnt_ties.keys():
                    cnt_ties[(att1, att2)] +=1
                else:
                    cnt_ties[(att2, att1)] +=1
            else:
                cnt_ties[(att1, att2)] += 1
            
    density = {key: cnt_ties[key]/total for key,total in cnt_pairs.items()}

    
    return(density, cnt_pairs, cnt_ties)


def diadic_density_two_vars(variable1, variable2, G_input, reciprocal=False):
    G = G_input.copy()
    attr1 = nx.get_node_attributes(G,variable1)
    attr2 = nx.get_node_attributes(G,variable2)
    
    ### Remove nodes that do not have the attributes
    nodes_rem = [key for key,val in attr1.items() if pd.isna(val) or pd.isna(attr2[key])]
    G.remove_nodes_from(nodes_rem)    
    attr1 = nx.get_node_attributes(G,variable1)
    attr2 = nx.get_node_attributes(G,variable2)

    cnt = Counter(zip(list([x for x in attr1.values() if not x=="Unknown"]),
                      list([x for x in attr2.values() if not x=="Unknown"])))
    
    ### Calculates total number of possible dyads for each pair of attributes
    cnt_pairs = {}
    for key,val in cnt.items():
        for key2, val2 in cnt.items():
            if reciprocal:
                if not (key2,key) in cnt_pairs.keys():
                    cnt_pairs[(key, key2)] = val*val2 if key!=key2 else (val*(val-1)/2)
            else:
                cnt_pairs[(key, key2)] = val*val2 if key!= key2 else val*(val2-1)
    
    ### Counts how many dyads in the network are between nodes with each possible attribute value
    cnt_ties = {key:0 for key in cnt_pairs.keys()}
    for edge in tqdm(G.edges()):
        att11 = G.nodes[edge[0]][variable1] 
        att12 = G.nodes[edge[0]][variable2]
        att21 = G.nodes[edge[1]][variable1]
        att22 = G.nodes[edge[1]][variable2]
        if att11 != "Unknown" and att12 != "Unknown":
            if att21 != "Unknown" and att22 != "Unknown":
                if reciprocal:
                    if ((att11, att12),(att21, att22)) in cnt_ties.keys():
                        cnt_ties[((att11, att12),(att21, att22))] +=1
                    else:
                        cnt_ties[((att21, att22),(att11, att12))] +=1
                else:
                    cnt_ties[((att11, att12), (att21, att22))] += 1
    
    density = {key:cnt_ties[key]/total for key,total in cnt_pairs.items()}
    return (density, cnt_pairs, cnt_ties)

### Transforms the output of the previous function into a table/dataframe
def get_table(d, vals = None, reciprocal=False):
    if vals == None:
        pairs = list(d.keys())
        vals = pd.unique([pair[0] for pair in pairs])
    df = pd.DataFrame(columns = vals, index= vals)
    for pair, val in d.items():
        df.loc[df.index == pair[0], df.columns == pair[1]] =  val
        if reciprocal:
            df.loc[pair[1], pair[0]] = val
    return(df)


#### calculates the distances between a random sample of dyads (G needs to have 0-len(G) indices)
def generate_random_distance(G, n, workers = 10):
    # Making random samples of the dinominator
    node1 = choices(list(G.nodes()), k = n)
    node2 = choices(list(G.nodes()), k = n)
    random_nodes = list(zip(node1, node2))
    
    #first get a dictionary of distance keyed by pair of nodes
    lat_d = nx.get_node_attributes(G,"lat")
    lon_d = nx.get_node_attributes(G,"lon")

    random_distance={}
    for pair in tqdm(random_nodes):
        coord1 = (lat_d[pair[0]], lon_d[pair[0]])
        coord2 = (lat_d[pair[1]], lon_d[pair[1]])
        dist = great_circle(coord1, coord2).km
        random_distance[pair]=dist
    return random_distance
    
### bin the distance and return a list of dictionary and counts of the binned distance
def bin_distance(distance, bins):
    #     #creat empty dicts according to number of bins, each dict in the list represents 
    dist_list = [dict() for number in range(len(bins))]
    cnt = {b:0 for b in bins}
    bin_lim = [x[0] for x in bins]
    #iterating through the distance dict
    for pair, dist in distance.items():
        if not dist:
            continue
        #check which bin the distance fits
        ind = bisect.bisect_left(bin_lim, dist)
        dist_list[ind-1][pair] = dist
        cnt[bins[ind-1]] += 1 
    
    return (dist_list, cnt)
        
    
def diadic_density_bin_distance(dem, binned_random_distance, binned_distance, G, factor = None, reciprocal=False):
    
    print("###########diadic_density_bin_distance##########")
    
    ### Calculates total number of possible dyads for each pair of attributes
    print("###########Calculates total number of possible dyad##########")
    cnt_pairs = {}
    #we use hte random distance here, because it is computationally impossible to 
    #calculate the possible pairs in the entire network.
    for pair, dist in binned_random_distance.items():
        #check if node exist in network
        if not G.has_node(pair[0]) or not G.has_node(pair[1]):
            # print("does not have node")
            continue
        att1 = G.nodes[pair[0]][dem]
        att2 = G.nodes[pair[1]][dem]
        if not pd.isna(att1) and not pd.isna(att2) and att1 != "Unknown" and att2 != "Unknown":
            key = (att1, att2) 
            if key in cnt_pairs.keys():
                # if it is reciprocal, we need to check if the reverse key is already in the list
                # then add one if NOT exist.
                if reciprocal:
                    if not (att2, att1) in cnt_pairs.keys():
                        cnt_pairs[key] += 1
                else:
                    cnt_pairs[key] += 1
            else:
                if reciprocal:
                    if not (att2, att1) in cnt_pairs.keys():
                        cnt_pairs[key] = 1
                else:
                    cnt_pairs[key] = 1
    
    ### Counts how many dyads in the network are between nodes with each possible attribute value
    
    cnt_ties = {key:0 for key in cnt_pairs.keys()}
    print("###########Counts how many dyads in the network##########")
    for edge, d in binned_distance.items():
        #check if node exist in network
        if not (G.has_node(edge[0]) and G.has_node(edge[1])):
            continue
        att1 = G.nodes[edge[0]][dem]
        att2 = G.nodes[edge[1]][dem]
        key = (att1, att2)
        if key in cnt_ties.keys():
            cnt_ties[(att1, att2)] += 1
        else:
            if reciprocal:
                cnt_ties[(att2, att1)] +=1
    
    if factor == None:
        factor = (100/len(G))
    density = {key:(cnt_ties[key]/total)*factor for key,total in cnt_pairs.items()}
    return(density, cnt_pairs, cnt_ties)

def generate_table_by_distance(dem, G_input, distance=None, random_distance=None, bins=None, factor = None, vals = None, reciprocal=False):
    
    print("###########generate_table_by_distance##########")
    
    print("Initialization")
    
    if not distance or not random_distance:
        G = G_input.copy()
        # this subsets the node keys that does not has na for coords and also filters out HI and Ak
        lat = nx.get_node_attributes(G,"lat")
        lon = nx.get_node_attributes(G,"lon")
        # this subsets the node keys that does not has na for coords and also filters out HI and Ak
        nodes_rem = [key for key,val in lat.items() if pd.isna(val) or pd.isna(lon[key]) or G.nodes()[key]['state'] in ['HI', 'AK']]
        G.remove_nodes_from(nodes_rem)
        G = nx.convert_node_labels_to_integers(G)
        distance=nx.get_edge_attributes(G,'distance')
        random_distance = generate_random_distance(G, len(G)*100)
    else:
        G = G_input
    
    print("Binning")
    ##bin them
    if bins == None:
        bins = [(0,20),(20,100),(100,500),(500,1000),(1000,2000), (2000, 5000)]
    binned_random_distance = bin_distance(random_distance, bins)[0]
    binned_distance = bin_distance(distance, bins)[0]
    
    print("Getting density tables")
    ##getting density tables
    table_list = []
    for i in range(len(bins)):
        print("Bin ", i, " processed")
        result = diadic_density_bin_distance(dem, binned_random_distance[i], binned_distance[i], G, factor, reciprocal)
        table = get_table(result[0], vals, reciprocal=reciprocal)
        table_list.append(table)
    return table_list
    


###This function will generate one heatmap given a table
### norm=False for raw probabilities, norm="Row" for row-normalized, norm="Max" to normalize by maximum value, and norm="Prop" to normalize by diagonal
def genreate_heatmaps(dem, df, norm = "Prop", size=None, col_names=None, file_name=None, is_reciprocal=False, value_range=None, title = True, annotate = True, font_scale=1.5, xticklabels='auto', yticklabels='auto', labelpad=15, color="magma_r", show = False):
    df = df.astype(float)
    label = "Probability of a tie"    
    
    if col_names:
        df = df[col_names]
        df = df.reindex(col_names).reindex(columns=col_names)
        
    if norm == "Row":
        df = df.div(df.sum(axis=1), axis = 0)
        label = "Row-normalized probability of a tie"
        
    if norm == "Prop":
        df = df.div(np.diag(df), axis = 0)
        label = "Probability of a tie relative to homophily"
        
    if norm == "Max":
        df = df/df.values.max()
        label = "Probability of a tie relative to maximum"
        
    if norm == False:
        label = "Probability of a tie"
        
    if size:
        fig, ax = plt.subplots(1,1, figsize = size)
    else:
        fig, ax = plt.subplots(1,1)
        
    # color_camp = sns.color_palette(color)
    if font_scale:
        sns.set(font_scale=font_scale) 

    if value_range:
        ax = sns.heatmap(df, annot = annotate, cmap=color, vmin = value_range[0], vmax = value_range[1],xticklabels=xticklabels, yticklabels=yticklabels) #cbar_kws={'label': label},cmap="YlGnBu",  
    else:
        ax = sns.heatmap(df, annot = annotate, cmap=color, xticklabels=xticklabels, yticklabels=yticklabels)
    
    
    # change distance of label from color bar
    cbar = ax.collections[0].colorbar
    cbar.set_label(label, labelpad=labelpad)
    
    if not is_reciprocal:
        ax.set_xlabel("Followed", fontdict = {'fontweight': "bold"}, labelpad=labelpad);
        ax.set_ylabel("Follower", fontdict = {'fontweight': "bold"}, labelpad=labelpad);
        
    else:
        ax.set_ylabel("Reference group", fontdict = {'fontweight': "bold"}, labelpad=labelpad);
        ax.set_xlabel("Out-group", fontdict = {'fontweight': "bold"}, labelpad=labelpad);

    if title: 
        ax.set_title("By "+dem, pad = 20, fontdict = {'fontweight': "bold"});
        if is_reciprocal:
            ax.set_title("Reciprocal edges by "+dem, pad = 20, fontdict = {'fontweight': "bold"});
        
    if file_name:
        fig.savefig(file_name,bbox_inches='tight', dpi=300, facecolor='white', transparent=False)
    if show:
        fig.show()
        
    #plt.close()

    return fig, ax



def genreate_heatmaps_by_distance(dem, df_list, index_list=None, is_row_normal=True, bins=None , unit = "km", sup = "by distance", is_reciprocal=False, file_name = None, size=(20,13), title = True, annotate = True, font_scale=1.5, xticklabels='auto', yticklabels='auto', labelpad=15, color="magma_r", show = False):
   
    label = "Probability of a tie"
    ##make sure the color scale are the same
    min_value = []
    max_value = []
    df_list_cp = copy.deepcopy(df_list)
    for i in range(len(df_list_cp)):
        if is_row_normal:
            df_list_cp[i] = df_list_cp[i].div(df_list_cp[i].sum(axis=1), axis = 0)
            label = "Row-normalized probability of a tie"
            if index_list:
                df_list_cp[i] = df_list_cp[i].reindex(index_list).reindex(columns=index_list)
        min_value.append(df_list_cp[i].values.min())
        max_value.append(df_list_cp[i].values.max())
    vmin = min(min_value)
    vmax = max(max_value)

    ##plot the tables
    ## do things in a forloop, handle differnet number of bins
    
    #initialize subplot so it has 3 columns, and math.ceil(len(bins)/3)
    nrows=math.ceil(len(df_list_cp)/3)
    ncols=3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    if font_scale:
        sns.set(font_scale=font_scale) 
    
    if bins == None:
        bins = [(0,20),(20,100),(100,500),(500,1000),(1000,2000), (2000, 5000)]
       
    cbar_ax = fig.add_axes([.93, .3, .03, .4])
    
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx >= len(df_list_cp):
                break
            sns.heatmap(df_list_cp[idx].astype(float), ax = ax[i][j], annot=annotate, xticklabels= i==nrows-1, yticklabels= j==0, cmap=color, 
                        vmin=vmin, vmax=vmax, cbar =  i+j==3, cbar_ax = None if i+j<3 else cbar_ax)
            ax[i, j].set_title(str(bins[idx][0])+'-'+str(bins[idx][1])+ ' ' + unit, pad = 15)
            idx += 1
    
    cbar = ax[1,2].collections[0].colorbar
    cbar.set_label(label, labelpad=15)
    
    if title:
        plt.suptitle(dem + sup, fontsize = 20, fontweight = "bold") 
    if is_reciprocal:
        plt.suptitle('Reciprocal edges ' + dem + sup, fontsize = 20) 

    if file_name:
        fig.savefig(file_name,bbox_inches='tight', dpi=300, facecolor='white', transparent=False)
        
    if show:
        plt.show()
    return True


###This function
def distance_density(G, bin_width = 50, factor = 100, bins =False, random_dist = False, filtered_G = False):
    # create a bin dist/list with distance range 50
    if bins == False:
        bins_dist = [(x, x+bin_width) for x in range(0,4600, bin_width)]
    else:
        bins_dist = bins
     
    # this subsets the node keys that does not has na for coords and also filters out HI and Ak
    if not filtered_G:
        nodes_keep = [node[0] for node in G.nodes(data=True) if not (node[1]["state"] in ['HI', 'AK']) and not pd.isna(node[1]['lat']) and not pd.isna(node[1]['lon'])]
        G = G.subgraph(nodes_keep).copy()
    
    print("Length of filtered network = ", len(G))
    
    if random_dist == False:
        random_distance = generate_random_distance(G, len(G)*factor)
    else:
        random_distance = random_dist
    
    cnt = bin_distance(random_distance, bins_dist)
    cnt = cnt[1]
    
    multiplier = len(G)/factor
    cnt = {key:(val*multiplier) for key,val in cnt.items()}

    distance = nx.get_edge_attributes(G,'distance')
    cnt_ties = bin_distance(distance, bins_dist)
    cnt_ties = cnt_ties[1]    

    density = {key:(cnt_ties[key]/total) for key,total in cnt.items() if total>0}
    return density


# Percentage of ties of ego-networks of each category that are of each other category. For example, 
# percentage of people followed by African-Americans who are African-Americans and who are White. 
# Help get a sense of how information that people get on Twitter may vary following their demographics, 
# and thinking of consequences of the homophily patterns we see. \
# Unknown cetegories are ignored and not counted for percentage
def ego_percentage_by_category(G, dem, lab_convert = True, weights = False):

    if lab_convert:
        G = nx.convert_node_labels_to_integers(G)
    attr = nx.get_node_attributes(G,dem)

    ## loop through the categories and make seperate sets
    subsets = {}
        
    ##loop through the attr, 
    for i in  tqdm(range(G.number_of_nodes())):
        #skip the node if it does not have an ego network
        if not len(list(G.neighbors(i))):
            continue
        # if the node's demograph is not recorded, we don't consider it.
        if pd.isna(attr[i]) or attr[i] == "Unknown":
            continue
        
        ## looping through the ego network
        ## getting the dem of the neighbors of each node
        ## for directed network, this gives us the nodes that n points to
        ## in this case, the people that n follows on twitter
        ego_attr = [attr[n] for n in G.neighbors(i)]
        ego_cnt = Counter(ego_attr) 
       
        ##calculate the percentage
        total = sum(ego_cnt.values())           
        if total == 0:
            continue
            
        if not weights == False:
            weighted_vals = {key:(val*weights[key]) for key,val in ego_cnt.items() if not key == "Unknown"}
            total_weighted = sum(weighted_vals.values())
            ego_density = {k:(val/total_weighted) for k,val in weighted_vals.items()}
        else :
            ego_density = {k:(val/total) for k,val in ego_cnt.items()}
        ego_density["Total_ties"] = total  
                
        self_category = attr[i]
        if not self_category in subsets.keys():
            subsets[self_category] = []
        subsets[self_category].append(ego_density)
        
        
    for c, d in subsets.items():
        df = pd.DataFrame(d)
        subsets[c] = df.fillna(0)
    ##return a table
    return subsets

## Use this function to get the stats table for the ego_percentage_by_category funciton
def get_stats(dem_percentage):
    categories = dem_percentage.keys()
    mean_df = pd.DataFrame(columns = categories, index= categories)
    std_df = pd.DataFrame(columns = categories, index= categories)
    median_df = pd.DataFrame(columns = categories, index = categories)
    for c, d in dem_percentage.items():
        for c2 in categories:
            c_mean = dem_percentage[c][c2].mean()
            c_std = dem_percentage[c][c2].std()
            c_med = dem_percentage[c][c2].median()
            mean_df.loc[c, c2] = c_mean
            std_df.loc[c,c2] = c_std
            median_df.loc[c,c2] = c_med
    return mean_df, std_df, median_df

#Percentage of ties of each ego network that are at different distances/population in the radius. 
## maybe smaller bins
## get median, average of distance, meam
def ego_percentage_by_distance(G, bins=[(0,500),(500,1000),(1000,5000)], lab_convert = True, rad_mod = False):

    if lab_convert:
        G = nx.convert_node_labels_to_integers(G)
        
    if not rad_mod:
        distance = nx.get_edge_attributes(G,'distance')
    else:
        distance = nx.get_edge_attributes(G, 'radiation_pop')
    #create empty list for the results
    ego_percentage_list = [] 
    ## looping through the ego network
    
    pairs_inc  = set(distance.keys())
    for ego in tqdm(range(G.number_of_nodes())):
        # skip if ego does not have neighbors
        if not len(list(G.neighbors(ego))):
            continue
        
        ## getting distance from ego to neighbors if distance exist
        neighbor_distance = {(ego, neighbor): distance[(ego,neighbor)] for neighbor in G.neighbors(ego) if 
                             (ego,neighbor) in pairs_inc and not pd.isna(distance[(ego,neighbor)])}
        ## bin distance, get the distance bin and counts
        binned = bin_distance(neighbor_distance, bins)[1]
        
        ## return the percentage dictionary
        total = len(neighbor_distance)
        if total <= 0:
            continue
        
        distance_percentage = {b:0 for b in bins}
        for b, cnt in binned.items():
            distance_percentage[b] = cnt/total
        distance_percentage["Total_ties"] = total
            
        ego_percentage_list.append(distance_percentage)

    
    ##return a table of ego and percentage of each bin
    return pd.DataFrame(ego_percentage_list)