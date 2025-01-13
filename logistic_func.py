import numpy as np
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import pandas as pd
from random import choices
from geopy.distance import great_circle
from joblib import Parallel, delayed
import networkx as nx
from multiprocessing import Pool
import math

### gets sample of negative dyads that have population in radius value
def get_dyads_rad(G,rand_pop):
    pos_dyads = set([edge for edge in G.edges()])
    neg_dyads = set(rand_pop.keys()).difference(pos_dyads)
    return(list(neg_dyads), list(pos_dyads))

### relies on having nodes numbered 1 to n
def get_random_dyads(G, n, nodes=False):
    if nodes == False:
        nodes = range(len(G))
    node1 = choices(nodes, k =n)
    node2 = choices(nodes, k =n)
    neg_dyads = [(n1, n2) for n1,n2 in zip(node1,node2)]
    neg_dyads = list(set(neg_dyads))
    pos_dyads = [edge for edge in G.edges()]
    lookup = set(pos_dyads)
    neg_dyads = [d for d in neg_dyads if not d in lookup]
    return(neg_dyads, pos_dyads)


def get_distance(node1, node2):
    coord1 = (node1['lat'], node1['lon'])
    coord2 = (node2['lat'], node2['lon'])
    dist = great_circle(coord1, coord2).km
    return(dist)
     
def get_density_diff(node1, node2):
    if node1["density_deciles"] == "90-100" and node2["density_deciles"] == "90-100":
        return(1)
    if node1["density_deciles"] == "0-10" and node2["density_deciles"] == "0-10":
        return(2)
    if node2["density_deciles"] == "90-100":
        return(3)
    else:
        return(0)
    

def is_na(node):
    if  (pd.isna(node['age']) or node['age']>114 
        or pd.isna(node['party_id']) 
        or pd.isna(node['race']) or pd.isna(node['gender']) 
        or node['gender']=="Unknown"
        or node['race']=="Unknown"
        or pd.isna(node["ruca_urbanicity"])
        or pd.isna(node["density_deciles"]) or pd.isna(node['state'])) :
        return(True)
    else:
        return(False)

def get_feature_hom(node1, node2, attr):
    if (pd.isna(node1[attr]) or pd.isna(node2[attr])):
        return("NA")
    if node1[attr] == node2[attr]:
        Hom = node1[attr]
    if node1[attr] != node2[attr]:
        Hom = 0
    return(Hom)

def get_feature_dyad(arg):
    node1 = arg[0]
    node2 = arg[1]    
    get_dist = arg[2]

    if is_na(node1) or is_na(node2):            
        return("NA")
    
    if get_dist:
        distance = get_distance(node1, node2)
    
    age_diff = np.abs(node2['age'] - node1['age'])
    age_diff_log = np.abs(math.log2(node2['age']) - math.log2(node1['age']))

    age_ego = node1['age']
    age_alter = node2['age']

    party_alter = node2["party_id"]
    party_ego = node1["party_id"]
    party_hom  = get_feature_hom(node1, node2, "party_id")
    
    party_diff = np.abs(node1["party_score"] - node2["party_score"])
    party_reg_alter = node2["party_reg"]
    party_reg_ego = node1["party_reg"]
    party_reg_hom  = get_feature_hom(node1, node2, "party_reg")
    
    race_alter = node2["race"]
    race_ego = node1["race"]
    race_hom  = get_feature_hom(node1, node2, "race")

    gender_alter = node2["gender"]
    gender_ego = node1["gender"]
    gender_hom= get_feature_hom(node1, node2, "gender")

    ruca_alter = node2["ruca_urbanicity"]
    ruca_ego = node1["ruca_urbanicity"]
    ruca_hom = get_feature_hom(node1, node2, "ruca_urbanicity")
    
    dens_diff = get_density_diff(node1,node2)
    
    same_state = 1 if node1['state'] == node2['state'] else 0 
    
    if get_dist:
        return(age_diff, age_diff_log, age_ego, age_alter, party_alter, party_ego, party_hom,
               party_reg_alter, party_reg_ego, party_reg_hom, party_diff, race_alter, race_ego, race_hom, 
               gender_alter, gender_ego, gender_hom, ruca_alter, ruca_ego, ruca_hom, dens_diff, same_state, distance)
    else:
        return(age_diff, age_diff_log, age_ego, age_alter, party_alter, party_ego, party_hom, 
               party_reg_alter, party_reg_ego, party_reg_hom, party_diff, race_alter, race_ego, race_hom, 
               gender_alter, gender_ego, gender_hom, ruca_alter, ruca_ego, ruca_hom, dens_diff, same_state)


def count_dyad_attr(node1, node2, attr, val):
    return(sum([node1[attr]==val, node2[attr]==val]))

def get_feature_dyad_rec(arg):
    node1 = arg[0]
    node2 = arg[1]
    get_dist = arg[2]

    if is_na(node1) or is_na(node2):            
        return("NA")
    
    if get_dist:
        distance = get_distance(node1, node2)
    
    age_diff = np.abs(node2['age'] - node1['age'])
    age_diff_log = np.abs(math.log2(node2['age']) - math.log2(node1['age']))
    age_sum = node1['age'] + node2['age']

    party_rep = count_dyad_attr(node1, node2, "party_id", "Republican")
    party_dem = count_dyad_attr(node1, node2, "party_id", "Democrat")
    party_hom  = get_feature_hom(node1, node2, "party_id")
    
    race_black = count_dyad_attr(node1, node2, "race", "African-American")
    race_hispanic = count_dyad_attr(node1, node2, "race", "Hispanic")
    race_asian = count_dyad_attr(node1, node2, "race", "Asian")
    race_other = count_dyad_attr(node1, node2, "race", "Other")
    race_native = count_dyad_attr(node1, node2, "race", "Native American")
    race_hom  = get_feature_hom(node1, node2, "race")

    gender_female = count_dyad_attr(node1, node2, "gender", "Female")
    gender_hom= get_feature_hom(node1, node2, "gender")
    
    ruca_rural = count_dyad_attr(node1, node2, "ruca_urbanicity", "small_town/rural")
    ruca_metropolitan = count_dyad_attr(node1, node2, "ruca_urbanicity", "metropolitan")
    ruca_hom = get_feature_hom(node1, node2, "ruca_urbanicity")
    
    dens_diff = get_density_diff_rec(node1,node2)
    
    same_state = 1 if node1['state'] == node2['state'] else 0 
    
    if get_dist:
        return(age_diff, age_diff_log, age_sum, party_hom, party_rep, party_dem,
               race_hom, race_black, race_hispanic, race_asian, race_other, race_native,
               gender_hom, gender_female,
               ruca_hom, ruca_rural, ruca_metropolitan, dens_diff, same_state, distance)
    else:
        return(age_diff, age_diff_log, age_sum, party_hom, party_rep, party_dem,
               race_hom, race_black, race_hispanic, race_asian, race_other, race_native,
               gender_hom, gender_female,
               ruca_hom, ruca_rural, ruca_metropolitan, dens_diff, same_state)

def get_features(neg_dyads, pos_dyads, G, radiation_pop, reciprocal = False, distance = False):
    
    print("Initializing data")
    
    dyads = neg_dyads + pos_dyads
    args = ([G.nodes[d[0]], G.nodes[d[1]], True if distance == True else False] for d in dyads)
 
    print("Getting edge features")    
    with Pool(30) as p:
        out = list(tqdm(p.imap(get_feature_dyad if not reciprocal else get_feature_dyad_rec, args, chunksize = 1000), total = len(dyads)))
    
    print("Filtering NAs")
    NAs = [i for i,el in enumerate(out) if el=="NA"]
    out = [row for row in out if row!="NA"]
    
    y = np.array([0 for i in range(len(neg_dyads))] + [1 for i in range(len(pos_dyads))])
    y = np.delete(y, NAs)
        
    print("Building dataframe")
    if not reciprocal:
        var_list = ["age_diff", "age_diff_log", "age_ego", "age_alter", "party_alter", "party_ego", "party_hom", 
                    "party_reg_alter", "party_reg_ego", "party_reg_hom", "party_diff", "race_alter", "race_ego", "race_hom",
                    "gender_alter", "gender_ego", "gender_hom", "ruca_alter", "ruca_ego", "ruca_hom", "dens_diff", "same_state"]
    else:
        var_list = ["age_diff", "age_diff_log", "age_sum", "party_hom", "party_rep", "party_dem",
                    "race_hom", "race_black", "race_hispanic", "race_asian", "race_other", "race_native",
           "gender_hom", "gender_female",
           "ruca_hom", "ruca_rural", "ruca_metropolitan", "dens_diff", "same_state"]  
    
    if distance == True:
        var_list.append("distance")
        
    X = pd.DataFrame(out, columns = var_list)    
    
    print("Adding edge population")
    dyads_left = pd.Series(dyads).iloc[list(set(range(len(dyads))).difference(NAs))]
    radiation_pop_vals = [radiation_pop.get(d) for d in dyads_left]
    X["radiation_pop"] = radiation_pop_vals

    print("Adding outcome and cleaning NA's")
    X['y'] = y    
    X.dropna(subset = ["radiation_pop"])    
    
    return(X)



