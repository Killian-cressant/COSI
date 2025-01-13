import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
#from test import *
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
import seaborn as sns
#from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data

from util.TimeDataset import TimeDataset


#from models.GDN import GDN

#from util.train import train
#from test  import test
#from util.evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


my_adjacency_matrix='/home/killian/Documents/Data/corr_adj.csv'

df_train=pd.read_csv('/home/killian/Documents/Data/dataset_test_full_server_1_preprocessed.csv', nrows=200)
columns_train=df_train.columns

df_val=pd.read_csv('/home/killian/Documents/Data/dataset_test_full_server_1_preprocessed.csv', nrows=200)
columns_val=df_val.columns

df_test=pd.read_csv('/home/killian/Documents/Data/dataset_test_full_server_1_preprocessed.csv', nrows=200)
columns_test=df_test.columns


#remove this value that is not in train but dont understand why ...
df_val.drop('node_vmstat_pgfault_server_1', axis=1, inplace=True)
df_test.drop('Unnamed: 0.1', axis=1 , inplace=True)
df_test.drop('node_vmstat_pgfault_server_1', axis=1 , inplace=True)
#removing unnamed
df_val=df_val.iloc[:,1:]
df_train=df_train.iloc[:,1:] 
df_test=df_test.iloc[:, 1:]
print(df_val.shape)
print(df_train.shape)
print(df_test.shape)
columns_train=df_train.columns
columns_val=df_val.columns
columns_test=df_test.columns



#### try to change the shape of the data

adj_25=np.genfromtxt(my_adjacency_matrix, delimiter=',')

def parcours_moit(m1):
    n=len(m1)
    for k in range(len(m1)):
        for j in range(len(m1[0])):
            if k+j+1<n:
                print(m1[j][j+k+1])

matrice = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#parcours_moit(matrice)



def fil_edges(Mat):
    edges_index=[]
    X1=[]
    X2=[]
    n=len(Mat)
    for k in range(len(Mat)):
        for j in range(len(Mat[0])):
            if k+j+1<n:
                if Mat[j][j+k+1]>0:
                    X1.append(j)
                    X2.append(j+k+1)
    edges_index.append(X1)
    edges_index.append(X2)
    edges_index=torch.tensor(edges_index)
    #edges_index=edges_index.float()
    return edges_index

edges_index=fil_edges(adj_25)
fc_edge_index = torch.tensor(edges_index, dtype = torch.long)


def return_matrix(edges):

    # Get the number of nodes (assuming nodes are zero-indexed and contiguous)
    num_nodes = torch.max(edges) + 1  # Assuming nodes start from 0

    # Create an empty adjacency matrix of zeros
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Fill the adjacency matrix based on the edges
    adj_matrix[edges[0], edges[1]] = 1  # Set 1 for each edge

    # If the graph is undirected, also fill the reverse edges
    # adj_matrix[edges[1], edges[0]] = 1  # Uncomment this for undirected graph

    #with simetric :

    adj_matrix=adj_matrix+adj_matrix.T
    
    print("done")
    return adj_matrix



plt.figure(figsize=(10, 8))
sns.heatmap(return_matrix(fc_edge_index), cmap='viridis')
plt.title('Adjacency matrix 25D')
plt.show()



batch=128
epoch=3 #100
slide_win=30  #peut-etre a modif aussi 15 c short #15
dim=64 #à changer
slide_stride=5 #?
comment=''
random_seed=0
out_layer_num=1
out_layer_inter_dim=256
decay=0 #?
val_ratio=0.1 # pas sur idée ce que c
topk=20 #normalement inutil
dataset='5G3E'
report = 'best' #jsp ce que c
device='cpu' #'cida' if GPU available but here CPU
load_model_path=''


train_config = {
    'batch': batch,
    'epoch': epoch,
    'slide_win': slide_win,
    'dim': dim,
    'slide_stride': slide_stride,
    'comment': comment,
    'seed': random_seed,
    'out_layer_num': out_layer_num,
    'out_layer_inter_dim': out_layer_inter_dim,
    'decay': decay,
    'val_ratio': val_ratio,
    'topk': topk,
}

env_config={
    'save_path': '',
    'dataset': dataset,
    'report': report,
    'device': device,
    'load_model_path': load_model_path
}







class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig=df_train
        test_orig=df_test #pour l'instant je met un val au lieu du test mais osef on ira pas jusque la
        #train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        #test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig
        print('loaded datasets correctly')

        #not useful for me but keep if training one day on wadi dataset
        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        #feature_map = get_feature_map(dataset)
        feature_map=df_train.columns#normaly it is just column names
        #fc_struc = get_fc_graph_struc(dataset) #pour moi le graph structure c la adjacency matrix
        fc_struc=np.genfromtxt(my_adjacency_matrix, delimiter=',')

        set_device(env_config['device'])
        self.device = get_device()
        #build_loc_net seems returning the edge_index so
        fc_edge_index=edges_index
        #fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map
        print("construction dataset")
        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())
        print("data construction done")

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])



        print("get loader done")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)
        print("dataloader done")

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)
        



    def run(self):

        for x, labels, attack_labels, edge_index in self.train_dataloader:
            plt.figure(figsize=(10,8))
            sns.heatmap(return_matrix(edge_index), cmap='viridis')
            plt.show()

            

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            #print(model_save_path)
            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,  #replace val_dataloader by test just to see if it is good now
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m-%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/test_result.csv'#{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":


    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    print("starting initialize main")
    print("="*80)
    main = Main(train_config, env_config, debug=False)
    print("Main initilized")
    print("-"*80)
    main.run()