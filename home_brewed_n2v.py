import torch 
import torch.nn.functional as F

from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils.num_nodes import maybe_num_nodes

'''
Does a biased random walk based on edge weight
'''
class GuidedNode2Vec(Node2Vec):
    def __init__(self, edge_index, edge_weight, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super(GuidedNode2Vec, self).__init__(
            edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q, 
            num_negative_samples, num_nodes, sparse
        )
        
        N = maybe_num_nodes(edge_index, num_nodes)
        self.adj = SparseTensor.from_edge_index(
            edge_index, 
            edge_attr=edge_weight, 
            sparse_sizes=(N, N)
        )
        self.adj = self.adj.to('cpu')
        
    '''
     Proceed with caution here. Tune the batch size for beest
     results. TODO do this automatically?
    '''
    def guided_walk(self, batch=None, batch_size=1028):
        if batch == None:
            batch = torch.tensor(range(self.adj.size(1)))
            
        batch = batch.repeat(self.walks_per_node)
        rw = []
        
        # This is to save memory if the adj matrix is huge
        # only look at batch_size rows of it at a time
        i = 0
        while(i*batch_size < batch.size()[0]):
            start = batch[i*batch_size:(i+1)*batch_size]
            
            # Convert to column 
            walk = [start.view(start.size()[0], 1)]
            
            for step in range(self.walk_length+1):
                prev = walk[-1]
                prev = prev.view(prev.size()[0])
                
                weights = self.adj[prev].to_dense()
                walk.append(torch.multinomial(weights, 1))
            
            # Cat all rws together
            rw.append(torch.cat(walk, dim=1))
            i += 1
        
        # Stack all random walks on top of each other
        rw = torch.cat(rw, dim=0)
            
        # Cut and paste from the pos_samples method to slice rws into
        # context_size chunks for the skip gram
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        
        return torch.cat(walks, dim=0)
    
    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        
        return self.guided_walk(batch=batch), self.neg_sample(batch)
        

    
    