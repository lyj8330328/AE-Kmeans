import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
class AutoEncoder():
    """ Auto Encoder  
    layer      1     2    ...    ...    L-1    L
      W        0     1    ...    ...    L-2
      B        0     1    ...    ...    L-2
      Z              0     1     ...    L-3    L-2
      A              0     1     ...    L-3    L-2
    """
    
    def __init__(self, X, Y, nNodes):
        # training samples
        self.X = X
        self.Y = Y
        # number of samples
        self.M = len(self.X)
        # layers of networks
        self.nLayers = len(nNodes)
        # nodes at layers
        self.nNodes = nNodes
        # parameters of networks
        self.W = list()
        self.B = list()
        self.dW = list()
        self.dB = list()
        self.A = list()
        self.Z = list()
        self.delta = list()
        for iLayer in range(self.nLayers - 1):
            self.W.append( np.random.rand(nNodes[iLayer]*nNodes[iLayer+1]).reshape(nNodes[iLayer],nNodes[iLayer+1]) ) 
            self.B.append( np.random.rand(nNodes[iLayer+1]) )
            self.dW.append( np.zeros([nNodes[iLayer], nNodes[iLayer+1]]) )
            self.dB.append( np.zeros(nNodes[iLayer+1]) )
            self.A.append( np.zeros(nNodes[iLayer+1]) )
            self.Z.append( np.zeros(nNodes[iLayer+1]) )
            self.delta.append( np.zeros(nNodes[iLayer+1]) )
            
        # value of cost function
        self.Jw = 0.0
        # active function (logistic function)
        self.sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))
        # learning rate 1.2
        self.alpha = 2.5
        # steps of iteration 30000
        self.steps = 10000
        
    def BackPropAlgorithm(self):
        # clear values
        self.Jw -= self.Jw
        for iLayer in range(self.nLayers-1):
            self.dW[iLayer] -= self.dW[iLayer]
            self.dB[iLayer] -= self.dB[iLayer]
        # propagation (iteration over M samples)    
        for i in range(self.M):
            # Forward propagation
            for iLayer in range(self.nLayers - 1):
                if iLayer==0: # first layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer-1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])            
            # Back propagation
            for iLayer in range(self.nLayers - 1)[::-1]: # reserve
                if iLayer==self.nLayers-2:# last layer
                    self.delta[iLayer] = -(self.X[i] - self.A[iLayer]) * (self.A[iLayer]*(1-self.A[iLayer]))
                    self.Jw += np.dot(self.Y[i] - self.A[iLayer], self.Y[i] - self.A[iLayer])/self.M
                else:
                    self.delta[iLayer] = np.dot(self.W[iLayer].T, self.delta[iLayer+1]) * (self.A[iLayer]*(1-self.A[iLayer]))
                # calculate dW and dB 
                if iLayer==0:
                    self.dW[iLayer] += self.X[i][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                else:
                    self.dW[iLayer] += self.A[iLayer-1][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                self.dB[iLayer] += self.delta[iLayer] 
        # update
        for iLayer in range(self.nLayers-1):
            self.W[iLayer] -= (self.alpha/self.M)*self.dW[iLayer]
            self.B[iLayer] -= (self.alpha/self.M)*self.dB[iLayer]
        
    def PlainAutoEncoder(self):
        for i in range(self.steps):
            self.BackPropAlgorithm()
            #print("step:%d" % i, "Jw=%f" % self.Jw)
    
    def ValidateAutoEncoder(self):
        data=[]
        for i in range(self.M):
            print(self.X[i])
            for iLayer in range(self.nLayers - 1):
                if iLayer==0: # input layer
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer-1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])
                print("\t layer=%d" % iLayer, self.A[iLayer])
                if iLayer==0:
                    s=self.A[iLayer]
                    ss=s.tolist()
                    data.append(ss)
        return data

#加载数据


datas = []
f = open('数据.txt','r')
for line in f.readlines():
    temp = []
    ss = line.replace('\n', '').split('\t')
    for i in range(len(ss)):
        temp.append(int(ss[i]))
    if i == len(ss)-1:
        datas.append(temp)
f.close()

scores=[]
for i in range(10):
    xx = np.array(datas)
    nNodes = np.array([9, 5, 9])
    ae2 = AutoEncoder(xx, xx, nNodes)
    ae2.PlainAutoEncoder()
    data = ae2.ValidateAutoEncoder()

    x = np.array(data)
    nNodes = np.array([5, 2, 5])
    ae2 = AutoEncoder(x, x, nNodes)
    ae2.PlainAutoEncoder()
    data3 = ae2.ValidateAutoEncoder()
    x1 = np.array(data3)
    xx = preprocessing.scale(x1)
    kmeans_model = KMeans(n_clusters=3).fit(xx)
    score=metrics.silhouette_score(xx, kmeans_model.labels_)
    scores.append(score)
print('------------------------------AE-KMeans聚类---------------------------')
print('轮廓系数为:')
print(max(scores))

print('------------------------------普通KMeans聚类---------------------------')
#print('-----------------------邻接矩阵--------------------')
#print(datas)
x = np.array(datas)
kmeans_model = KMeans(n_clusters=3).fit(x)
#print('----------------------------------')
#print('inertia_为：')
#print(kmeans_model.inertia_)
score = metrics.silhouette_score(x, kmeans_model.labels_)
#print('----------------------------------')
print('轮廓系数为:')
print(score)
print('-----------------------------------------------------------------------')

