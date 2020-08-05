import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan as NA
#from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
#from sklearn import preprocessing
##from sklearn import linear_model,feature_selection
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
#import statsmodels.graphics.api as smg
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering

#####  Clustering  ####
'''
## load data
path='car_evaluation.csv'
names=['buying','maint','doors','persons','lug_boot','safety','acceptability']
dataframe=pd.read_csv(path,delimiter=',',names=names)

###  LABEL ENCODING

for ix in names:
    iy = dataframe[ix].unique()
    dataframe[ix].replace(iy, range(1,len(iy)+1), inplace = True)
dataframe.to_csv('car_eval_encoded.csv',index=False)
'''
#### LOAD ENCODED DATA
df=pd.read_csv('car_eval_encoded.csv')

### CALC INTERACTIONS
rfactors=['maint','doors','persons','lug_boot','safety','acceptability']

#for ix in range(0,len(rfactors)-1):
    #for iy in range(ix+1,len(rfactors)):
        #new_name=str(rfactors[ix]+'_'+rfactors[iy])
        #df[new_name]=df[rfactors[ix]]*df[rfactors[iy]]

## prepare data
Y=df['buying']
X=df[rfactors]

#Y=df.iloc[:,0]
#X=df.iloc[:,1:22]

'''
### K Means Cluster
model=KMeans(n_clusters=4,random_state=11)
model.fit(X)

### smeniame 1 ot fit na 0; 0 ot fit na 1,kakto waw faila
df['pred_buying']=np.choose(model.labels_,[1,2,3,4]).astype(np.int64)

## show results
print('Accuracy:',metrics.accuracy_score(df['buying'],df['pred_buying']))
print('Classification report:',metrics.classification_report(df['buying'],df['pred_buying']))

## plot results
labels_1 = ['low', 'med','high','vhigh']
plt.figure(figsize=(10,7))

#####  Side by Side Bar Chart   ####
# generate simple data
ix = df['pred_buying'].value_counts(sort=False)
iy = df['buying'].value_counts(sort=False)

pre=iy.values
post=ix.values
labels=['low', 'med','high','vhigh']
# the plot - left and right
width=0.4 # bar width
xlocs=np.arange(len(pre))
plt.bar(xlocs-width,pre,width,color='green',label='buying')
plt.bar(xlocs,post,width,color='#1f10ed',label='pred_buying')
# labels,grids,titles,save
plt.xticks(range(len(pre)),labels)

plt.legend(loc='best')
plt.ylabel('Count')
plt.suptitle('Sample Chart')
#plt.tight_layout()  #(pad=1)
plt.savefig('barchart.png',dpi=100)

#####  Finding the value of k

####   Elbow method

### K Means Cluster
K=range(1,10)

KM=[KMeans(n_clusters=k).fit(X) for k in K]
centroids=[k.cluster_centers_ for k in KM]

D_k=[cdist(X,cent,'euclidean') for cent in centroids]
cIdx=[np.argmin(D,axis=1) for D in D_k]
dist=[np.min(D,axis=1) for D in D_k]
avgWithinSS=[sum(d)/X.shape[0] for d in dist]
## Total with-in sum of square
wcss=[sum(d**2) for d in dist]
tss=sum(pdist(X)**2)/X.shape[0]
bss=tss-wcss
varExplained=bss/tss

adjRsq=[1-((1-varExplained[k-1])*(X.shape[0]-1)/(X.shape[0]-k-1)) for k in K]
print([adjRsq[k+1]-adjRsq[k] for k in range(0,8)])
kIdx=10-1
#### plot ####
kIdx=3
###  elbow curve
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot((K),(avgWithinSS),'b*-')
plt.plot(K[kIdx],avgWithinSS[kIdx],marker='o',markersize=12,markeredgewidth=2,markeredgecolor='r',markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Elbow for KMeans clustering')
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot((K),(adjRsq),'b*-')
plt.plot(K[kIdx],adjRsq[kIdx],marker='o',markersize=12,markeredgewidth=2,markeredgecolor='r',markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Persentage of variance explained adj')
plt.tight_layout()
plt.savefig('elbow_graphs.png')

########  Average Silhouette Method

score = []
for n_clusters in range(2,10):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score.append(silhouette_score(X, labels, metric='euclidean'))
# Set the size of the plot
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(score)
plt.grid(True)
plt.ylabel("Silouette Score")
plt.xlabel("k")
plt.title("Silouette for K-means")
# Initialize the clusterer with n_clusters value and a random generator
model = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=10)
model.fit_predict(X)
#### smeniame 1 ot fit na 0; 0 ot fit na 1,kakto waw faila
#df['pred_buying']=np.choose(model.labels_,[1,2,3,4]).astype(np.int64)
cluster_labels = np.unique(model.labels_)
n_clusters = cluster_labels.shape[0]
# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(X, model.labels_)
plt.subplot(1, 2, 2)
# Get spectral values for colormap.
cmap = cm.get_cmap("Spectral")
y_lower, y_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[cluster_labels]
    c_silhouette_vals.sort()
    y_upper += len(c_silhouette_vals)
    color = cmap(float(i) / n_clusters +0.1)
    plt.barh(range(y_lower, y_upper), c_silhouette_vals, facecolor=color,edgecolor=color, alpha=0.7)
    yticks.append((y_lower + y_upper) / 2)
    y_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.yticks(yticks, cluster_labels+1)
# The vertical line for average silhouette score of all the values
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title("Silouette for K-means")
plt.savefig('Silouette_graphs.png')
'''
#####  Hierarchical Clustering
'''
# Agglomerative Cluster
model = AgglomerativeClustering(n_clusters=4)
# lets fit the model to the iris data set that we imported in Listing 3-49
model.fit(X)
### smeniame 1 ot fit na 0; 0 ot fit na 1,kakto waw faila
df['pred_buying']=np.choose(model.labels_,[1,2,3,4]).astype(np.int64)
print("Accuracy :", metrics.accuracy_score(df['buying'], df['pred_buying']))
'''



