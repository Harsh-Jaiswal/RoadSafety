# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import os
input_data_file = "bangalore-cas-alerts.csv"

SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 10, 12, 20
plt.rc('font', size=MEDIUM_SIZE)       
plt.rc('axes', titlesize=BIG_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE) 
plt.rc('ytick', labelsize=MEDIUM_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIG_SIZE)
my_colors = 'rgbkymc'

train_data = pd.read_csv(input_data_file)
train_data.head(10)
columns={
            "deviceCode_deviceCode" : "deviceCode",
            "deviceCode_location_latitude" : "latitude",
            "deviceCode_location_longitude" : "longitude",
            "deviceCode_location_wardName" : "wardName",
            "deviceCode_pyld_alarmType" : "alarmType",
            "deviceCode_pyld_speed" : "speed",
            "deviceCode_time_recordedTime_$date" : "recordedDateTime"
        }
train_data.rename(columns=columns, inplace=True)
train_data.drop_duplicates(keep=False,inplace=True) 
lis=[]
a=[]
for i in range(len(train_data)):
	a=[train_data.iloc[i,1],train_data.iloc[i,2]]
	lis.append(a)
# train_data.drop(train_data.loc[train_data['wardName']=="other"].index, inplace=True)
# train_data.drop(train_data.loc[train_data['wardName']=="Other"].index, inplace=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten

# lat_max = train_data.latitude.max()
# lat_min = train_data.latitude.min()
# print("Range of latitude:", lat_max, lat_min)

# lon_max = train_data.longitude.max()
# lon_min = train_data.longitude.min()

# epsilon = 0.01
# bound_box = [lon_min + epsilon, lon_max + epsilon, 
#              lat_min + epsilon, lat_max + epsilon]
# print("Range of longitude:", lon_max, lon_min)
from sklearn.cluster import KMeans
coordinates=  np.array(lis)
kmeans = KMeans(n_clusters = 10).fit(coordinates)
centroids = kmeans.cluster_centers_
pred_clusters = kmeans.predict(coordinates)
coordinates=  np.array(lis)
# x, y = kmeans2(whiten(coordinates), 15, iter = 50)  
print(centroids)
plt.scatter(coordinates[:,0], coordinates[:,1], c=pred_clusters)
#plt.tight_layout()
#plt.show()
s=20*4**1.5
x=np.array(centroids)
plt.scatter(x[:,0],x[:,1],marker="^",s=s,color='red')
plt.show()

from sklearn.cluster import KMeans
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(coordinates)
  labels = kmeans.labels_
  sil.append(silhouette_score(coordinates, labels, metric = 'euclidean'))
plt.plot(sil)
plt.show()



# bangalore_map_img = 'https://lh3.googleusercontent.com/np8igtYRrHpe7rvJwMzVhbyUZC4Npgx5fRznofRoLVhP6zcdBW9tfD5bC4FbF2ITctElCtBrOn7VH_qEBZMVoPrTFipBdodufT0QU1NeeQVyokMAKtvSHS9BfYMswXodz_IrkiZStg=w500-h664-no'

# cmap = plt.get_cmap("jet")
# # # im = plt.imshow(bangalore_map, extent=bound_box, zorder=0, 
# # #            cmap=cmap, interpolation='nearest', alpha=0.7)
# # plt.imshow(bangalore_map, extent=bound_box, zorder=0, 
# #            cmap=cmap, interpolation='nearest', alpha=0.7)
# #cbar = plt.colorbar(im, fraction=0.05, pad=0.02)
# img = plt.imread(bangalore_map_img)
# fig, ax = plt.subplots()
# ax.imshow(img)
# fig, ax = plt.subplots()