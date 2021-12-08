import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import pygraph as pg
import kernel
import torch.utils.dlpack
from time import perf_counter

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

#pygraph code starts
def memoryview_to_np(memview, nebr_dt):
    arr = np.array(memview, copy=False)
    #a = arr.view(nebr_dt).reshape(nebr_reader.get_degree())
    a = arr.view(nebr_dt)
    return a

edge_dt = np.dtype([('src', np.int32), ('dst', np.int32), ('val',np.float32)])
flags = pg.enumGraph.eUdir
outdir = ""
graph  = pg.init(1,1, outdir, 1, 2) # Indicate one pgraph, and one vertex type
tid0 = graph.init_vertex_type(adj.shape[0], False, "gtype"); # initiate the vertex type
pgraph = graph.create_schema(flags, tid0, "friend", edge_dt); #initiate the pgraph

dd = np.zeros(10000, edge_dt)

tempGraph = adj.coalesce()
rowList = tempGraph.indices()[1].tolist()
colList = tempGraph.indices()[0].tolist()
valList = tempGraph.values().tolist()
feat = features.tolist()
feattensor = torch.tensor(feat, dtype=torch.float32)

edge_count = 0
for i in  range(0,len(rowList)):
    dd[edge_count]= (rowList[i], colList[i], valList[i])
    edge_count += 1
    if (edge_count == 10000):  
        pgraph.add_edges(dd, edge_count)
        edge_count = 0

pgraph.add_edges(dd, edge_count)
pgraph.wait()

offset_csr1, offset_csc1, nebrs_csr1, nebrs_csc1 = pg.create_csr_view(pgraph)
offset_dt = np.dtype([('offset', np.int32)])
csr_dt =  np.dtype([('dst', np.int32),('val',np.float32)])

offset_csr = memoryview_to_np(offset_csr1, offset_dt)
offset_csc = memoryview_to_np(offset_csc1, offset_dt)
nebrs_csr  = memoryview_to_np(nebrs_csr1, csr_dt)
nebrs_csc  = memoryview_to_np(nebrs_csc1, csr_dt)

flag = 0
G = kernel.init_graph(offset_csr, nebrs_csr, offset_csc, nebrs_csc, flag, adj.shape[0])

X_dl = torch.utils.dlpack.to_dlpack(feattensor)
res = torch.zeros(features.shape[0], features.shape[1])
res_dl = torch.utils.dlpack.to_dlpack(res)

#sgc_precompute with kernel
t = perf_counter()
for i in range(args.degree):
    kernel.spmm(G, X_dl, res_dl)
    if (i<args.degree-1):
        X_dl = torch.utils.dlpack.to_dlpack(res)
        res = torch.zeros(features.shape[0], features.shape[1])
        res_dl = torch.utils.dlpack.to_dlpack(res)

print("kernel spmm time: "+"{:.4f}s".format(perf_counter()-t)) 

if (args.cuda):
    res = res.to(device='cuda:0')
#pygraph code ends


model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC" or args.model == "nSGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("pytorch spmm api time: "+"{:.4f}s".format(precompute_time))

#uses the output feature from the kernel instead of the python api comment this out if you want the orginal
features = res

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

if args.model == "SGC" or args.model == "nSGC":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
