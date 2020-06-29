import torch 
import torch.nn.functional as F

from torch.nn.modules import Linear

def n2v_trainer(model, N, epochs=500, early_stopping=True, 
                    patience=10, verbosity=1, lr=0.01):
        
        print("Training n2v")
        
        stopped_early = True
        loss_min=1000
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
        increase = 0
        state_dict_save = 'n2v_state_dict.model'
        
        loader = model.loader(batch_size=N, shuffle=True, num_workers=4)
        
        for epoch in range(epochs):
            total_loss = 0
            for pos, neg in loader:
                model.train()
                
                optimizer.zero_grad()
                loss = model.loss(pos, neg)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            total_loss /= len(loader)
            if total_loss > loss_min and early_stopping:
                increase+= 1
            else:
                if verbosity > 0:
                    print("===New Minimum loss===")
                    print('[%d] Loss: %.3f' % (epoch, loss))
                loss_min = loss
                increase=0
                torch.save(model.state_dict(), state_dict_save)
                
            if increase > patience:
                print("Early stopping!")
                stopped_early=True
                break
            
            if increase > patience//4 and lr > 5e-6:
                lr -= 0.00025
                lr = lr if lr > 5e-6 else 5e-6 # make sure not less than 5e-6
                print('LR decay: New lr: %.6f' % lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr
            
            
        if stopped_early:
            print("Reloading best parameters!")
            model.load_state_dict(torch.load(state_dict_save))
            
        return model
    
def train_lr(data, embedder, epochs=500, early_stopping=True, 
            patience=10, verbosity=1, lr=0.01, out_feats=1):
    
    # Haddamard embedded source with embedded dst to produce positive samples
    edge_combo_embedder = lambda x : embedder(x[0]) * embedder(x[1])
    
    # Generate random edges (assume no overlap for now)
    neg_edge_generator = lambda x : torch.randint(0, data.num_nodes-1, data.edge_index.size())

    pos_edge = edge_combo_embedder(data.edge_index)
    y = torch.zeros((pos_edge.size()[0] * 2, out_feats), dtype=torch.float)
    y[:, :pos_edge.size()[1]] = 1

    model = LogRegress(embedder.embedding_dim, out_feats)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state_dict_save = 'model.out'
    stopped_early = False

    loss_min = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        optimizer.zero_grad()
        
        neg_edge = edge_combo_embedder(neg_edge_generator(0)) 
        X = torch.cat([pos_edge, neg_edge], dim=0)
        
        loss = F.mse_loss(
            model(X),
            y
        )
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss += loss.item()
        if total_loss > loss_min and early_stopping:
            increase+= 1
        else:
            if verbosity > 0:
                print("===New Minimum loss===")
                print('[%d] Loss: %.3f' % (epoch, loss))
            loss_min = loss
            increase=0
            torch.save(model.state_dict(), state_dict_save)
            
        if increase > patience:
            print("Early stopping!")
            stopped_early=True
            break
        
        if increase > patience//4 and lr > 5e-6:
            lr -= 0.00025
            lr = lr if lr > 5e-6 else 5e-6 # make sure not less than 5e-6
            print('LR decay: New lr: %.6f' % lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            
    if stopped_early:
        print("Reloading best parameters!")
        model.load_state_dict(torch.load(state_dict_save))
        
    return model


class LogRegress(torch.nn.Module):
    def __init__(self, num_feats, out_feats):
        super().__init__()
        self.w = Linear(num_feats, out_feats)
    
    def forward(self, X):
        return torch.sigmoid(self.w(X))
    
