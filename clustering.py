# =============================================================================
# This File is responsible for doing Primary clustering
# =============================================================================

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances

path = os.path.abspath(os.path.dirname(__file__))  
pickle_path = os.path.join(path, "Pickled Data")

with open(pickle_path+"/word_vector.pickle","rb") as pickle_in:
    word_vector = pickle.load(pickle_in)

with open(pickle_path+"/unique_Wordset.pickle","rb") as pickle_in:
    Wordset = pickle.load(pickle_in)

with open(pickle_path+"/word_Frequency.pickle","rb") as pickle_in:
    word_freq = pickle.load(pickle_in)
    
with open(pickle_path+"/TF_values.pickle","rb") as pickle_in:
    tf= pickle.load(pickle_in)


newDF=pd.DataFrame(tf)    
num_c =25
totalWords =[index for index,rows in newDF.iterrows()]
 

#Elbow method to Find the Appropriate Clusters
"""
distortions = []

K = range(1,50)
for k in K:
    cluster = KMeans(n_clusters =k).fit(newDF)
    cluster.fit(newDF)
    #cluster = AgglomerativeClustering(n_clusters = num_c,affinity='euclidean',linkage='ward').fit_predict(newDF)
    distortions.append(sum(np.min(cdist(newDF, cluster.cluster_centers_, 'euclidean'), axis=1)) / newDF.shape[0])

#principalDF['cluster']=cluster
 

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""

#Applying k means Clutering
#cluster = AgglomerativeClustering(n_clusters = 9,affinity='euclidean',linkage='ward').fit_predict(newDF)
cluster = KMeans(n_clusters =9,init='k-means++', n_init=100, max_iter=1000, tol=0.0006,random_state=100).fit_predict(newDF)
#cluster.fit(newDF)
#labels = cluster.labels_
#value=metrics.silhouette_score(newDF, cluster, metric='euclidean')
#print(value)
newDF['cluster']=cluster


#Applying PCA
pca = PCA(n_components= 2)
principalComp = pca.fit_transform(newDF)
principalDF = pd.DataFrame(data=principalComp,columns=['PC1','PC2'])
principalDF.index = totalWords
mergedDF=newDF.merge(principalDF ,left_index=True, right_index=True, how='inner')

#print(mergedDF)

#Plotting the Graph
"""
colors= [str(round(random.uniform(0, 1),3)) for x in range(11)]
LABEL_COLOR_MAP = {}
for value in range(11):
    LABEL_COLOR_MAP[value]=colors[value]
label_color = [LABEL_COLOR_MAP[l] for l in mergedDF['cluster']]
plt.scatter(mergedDF['PC1'],mergedDF['PC2'], c=label_color)
    
plt.xlabel('PCA1') 
plt.ylabel('PCA2')
plt.show()
"""
#Dictionary for clusters
clusterDict = {}
 
for index, row in newDF.iterrows():
    if row['cluster'] in clusterDict:
        clusterDict[row['cluster']].append(index)
    elif row['cluster'] not in clusterDict:
        clusterDict[row['cluster']]=[index]

privacy =['protection','interests','licenses','protect','property','intellectual','notices','protections','authorization','obligations'] 

for clusters in clusterDict:
    clusterDict[clusters]=sorted(clusterDict[clusters])

for clus in clusterDict:
    print(clus,clusterDict[clus])
with open(pickle_path+"/K_MeansCluster.pickle","wb") as pickle_out:
    pickle.dump(clusterDict,pickle_out)
pickle_out.close()