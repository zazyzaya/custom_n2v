import os 
import sys 
import json 
import torch 

from torch_geometric.data import Data
from home_brewed_n2v import GuidedNode2Vec
from torch_geometric.nn.models import Node2Vec

from run_n2v import n2v_trainer, train_lr

DATA = '/mnt/raid0_24TB/datasets/pico/bro/'
DATES = ['2019-07-' + str(d) + '/' for d in [19, 20, 21]]

def parse_file(fname, db={}):
    if not fname.split('/')[-1].startswith('kerberos'):
        return db
    
    with open(fname, 'r') as f:
        logs = f.read()
        
    for log in logs.split('\n'):
        if len(log.strip()) == 0:
            continue
        
        d = json.loads(log)
        
        # Only log events within the network
        if 'client' in d:
            src, dst = d['client']+':'+d['id.orig_h'], d['service']
            
            if (src,dst) in db:
                db[(src,dst)] += 1
            else:
                db[(src,dst)] = 1
                
    return db

def read_all_files(days=DATES):
    db = {}
    
    for d in days:
        for f in os.listdir(DATA + d):
            db = parse_file(DATA + d + f, db=db)
            
    return db

def db_to_graph(db, node_map={}):
    if len(node_map) == 0:
        src,dst = zip(*list(db.keys()))
        nodes = list(set(src).union(set(dst)))
        node_map = dict([(n,v) for v,n in enumerate(nodes)])
     
    # Cant say much about new nodes we've never seen   
    else:
        ks = list(db.keys())
        for k in ks:
            if k[0] not in node_map or k[1] not in node_map:
                db.pop(k)
                
        src,dst = zip(*list(db.keys()))
        nodes = list(set(src).union(set(dst)))
        
    N = len(nodes)
    
    # Just go featureless 
    X = torch.eye(N,N, dtype=torch.float)
    
    # 2D src-dst mapping; flip for non-directed graph
    edge_index = torch.tensor([
        [node_map[n] for n in src] + [node_map[n] for n in dst],
        [node_map[n] for n in dst] + [node_map[n] for n in src]
    ])
    
    # 1D weight mapping corresponding to e_i indices
    edge_weight = torch.tensor(
        [db[edge] for edge in zip(src,dst)] * 2,
        dtype=torch.float
    )
    
    mapping = dict([(v,k) for k,v in node_map.items()])
    
    return Data(
        x=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mapping=mapping
    ), node_map
    
baseline, nm = db_to_graph(read_all_files([DATES[0]]))
attack, _ = db_to_graph(read_all_files(DATES[1:]), node_map=nm) 


#Default
if input('Use guided Node2Vec? (y/n) ') == 'n':
    print("Using default n2v")
    embedder = Node2Vec(
        baseline.edge_index,
        64,             # Embedding dimesion
        5,              # Walk len  
        3,              # Context size 
        sparse=True
    )

# Upgraded (?)
else:
    print("Using n2v biased by edge weight")
    embedder = GuidedNode2Vec(
        baseline.edge_index,
        baseline.edge_weight,
        64,             # Embedding dimesion
        5,              # Walk len  
        3,              # Context size 
        sparse=True
    )

n2v_trainer(
    embedder,
    baseline.x.size()[0],
    lr=0.05,
    patience=100,
    epochs=1000
)   

predictor = train_lr(
    baseline,
    embedder,
    patience=100,
    epochs=1000,
    out_feats=1
)

edge_combo_embedder = lambda x : embedder(x[0]) * embedder(x[1])

ei = attack.edge_index
risk = []
for i in range(ei.size()[1]):
    rating = predictor(
        edge_combo_embedder(ei[:, i])
    )

    src = baseline.mapping[ei[0, i].item()]
    dst = baseline.mapping[ei[1, i].item()]
    
    if ':' in dst:
        tmp = dst
        dst = src
        src = tmp 
    
    risk.append(("%s -> %s" % (src, dst), rating.mean().item()))
    

risk.sort(key=lambda x : x[1], reverse=True)
for r in risk:
    print(r[0] + ': ' + str(r[1]))