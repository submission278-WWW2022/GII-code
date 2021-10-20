from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from torch_geometric.datasets import Planetoid
import argparse


parser = argparse.ArgumentParser(description='Args for visualization')
parser.add_argument('-dataset', type=str, default="citeseer", help='dataset')
parser.add_argument('-rep_file', type=str, default="citeseer71.99799346923828.txt", help='representation file')
args, _ = parser.parse_known_args()

dataset_name = args.dataset     # choose the dataset
rep_file = args.rep_file      # choose the representations to visualize
rep_path = osp.join("embeddings", rep_file)
dataset = Planetoid("./data", dataset_name)[0]

y_true = dataset.y.numpy()
n_cluster = np.max(y_true) + 1
print("Number of clusters:", n_cluster)
data = np.loadtxt(rep_path)
# data = dataset.x.numpy().astype(np.float)        # raw features
color = y_true
color_label = np.array([''] * len(color))

if dataset_name == "cora":
    print("Dataset:", dataset_name)
    color_label[color==0] = "red"
    color_label[color==1] = "green"
    color_label[color==2] = "blue"
    color_label[color==3] = "k"
    color_label[color==4] = "c"
    color_label[color==5] = "black"
    color_label[color==6] = "y"

if dataset_name == "citeseer":
    print("Dataset:", dataset_name)
    color_label[color==0] = "k"
    color_label[color==1] = "green"
    color_label[color==2] = "blue"
    color_label[color==3] = "red"
    color_label[color==4] = "c"
    color_label[color==5] = "m"

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)

fig = plt.figure(figsize=(6, 4), dpi=300)
plt.subplot(1, 1, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_label, s = 8)
plt.axis('off')
fig.savefig("./fig.pdf", dpi=300, bbox_inches='tight')
plt.show()

