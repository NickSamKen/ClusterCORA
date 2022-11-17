import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ari
from sklearn.metrics import accuracy_score as acc
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
# <paperID> <word vocab>+ <class_label>
# use n_init for choosing starting centroid
f_edge = 'D:/Users/User/cora/cora.cites' #file location of the columns
f_nodes = 'D:/Users/User/cora/cora.content' #file location of the rows


C = 7 #initiate desired kmeans cluster number
V = 1433 #initiate vocabulary number
word_col_names = ["w{}".format(i) for i in range(V)] #creates words column
df_node = pd.read_csv(f_nodes, sep='\t', header=None, names=['paperid']+word_col_names+['label']) #creates the matrix
print(df_node)#displays the matrix

X = df_node[word_col_names].to_numpy() #turns the matrix into a numpy array
pca = PCA(30)#use pca
data = pca.fit_transform(X) #makes the numpy array fit the pca specifications
model = KMeans(n_clusters=C, random_state=None, init='random', n_init = 15, max_iter = 600)  #initiate the cluster with the desired parameters
model.fit(data) #input the new transformed kmeans into the kmeans model
print(model.labels_[:20]) #prints model label
yhats = model.labels_
np.unique(yhats)#list of the classes
yhats
print(plt.hist(yhats, bins='auto'))#prints the chart

y, y_idx = pd.factorize(df_node['label'])#factorize the df_nodes
y_mapping = {y_idx[k]: k for k in range(7)}#matching the index with the array count
print(plt.hist(y, bins='auto'))#chart y chart post factorzation
np.bincount(model.labels_[y==0])# shows the number of Docs for each cluser label
print(np.bincount(yhats[y==0])/np.bincount(yhats)) #true yhats over total yhats

from sklearn.preprocessing import normalize#increase the accuracy by using normalize
v = np.bincount(yhats[y==0])/np.bincount(yhats)
normalized_v = normalize(v[:,np.newaxis], axis=0).ravel()
hungarian = []#hungarian algorithm finds mapping between cluster numbers and gtlabels using linear_sum_assignment
for k in range(7):
    hungarian.append(np.bincount(yhats[y==k]))
_, hg_mapping = linear_sum_assignment(hungarian, maximize=True)
for k in y_mapping:#conversion ground truth labels
    masked_labels = model.labels_
    best_cls = np.argmax(
        np.bincount(yhats[y == y_mapping[k]]) / np.bincount(yhats)
    )
for m, m_name in ((nmi, 'NMI'), (ari, "ARI")):
    print(f'{m_name}:{m(y,yhats):.3f}')
acc([hg_mapping[i] for i in y], yhats) #printing accuracy
X_embedded = TSNE(n_components=2, verbose=1, perplexity=40).fit_transform(X)
centroids = model.cluster_centers_
plt.figure(figsize=(10,10))
uniq = np.unique(X_embedded)
sns.scatterplot(x = X_embedded[:,0], y = X_embedded[:,1], hue=y, legend='brief')