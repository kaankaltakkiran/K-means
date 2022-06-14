import numpy as np
import matplotlib.pyplot as plt

# veri seti
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')
    return data

# Öklid uzaklığı hesaplaması
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2)) 

# Belirli bir veri kümesi için bir K rasgele merkez kümesi oluşturun
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids

# k-kümeleme anlamına gelir 
def KMeans(dataSet,k):

    m = np.shape(dataSet)[0]  #satır sayısı 
    # İlk sütun, örneğin ait olduğu kümeyi saklar?
    # İkinci sütun, örneğin hatasını kümenin merkez noktasına kaydeder. 
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    # Adım 1 Merkezleri başlat
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False

        # Tüm örnekler üzerinde yineleme (satır sayısı) 
        for i in range(m):
            minDist = 100.0
            minIndex = -1

            # tüm centroidler üzerinde yineleme 
            # Adım 2 En yakın ağırlık merkezini bulun 
            for j in range(k):
                # Numuneden merkeze Öklid mesafesini hesaplayın 
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # Adım 3: Her bir örnek satırının ait olduğu kümeyi güncelleyin 
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #Adım 4: Merkezleri güncelleyin 
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # Küme sınıfının tüm puanlarını alın 
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # Bir matrisin satırlarının ortalamasını alın 

    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    if n != 2:
        print("veri iki boyutlu değil ")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k değeri çok büyük ")
        return 1

    # tüm örnekleri çiz 
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # merkezler çiz 
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i])

    plt.show()
dataSet = loadDataSet("test.txt")
k =4
centroids,clusterAssment = KMeans(dataSet,k)

showCluster(dataSet,k,centroids,clusterAssment)