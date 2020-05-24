import time
import numpy as np
import math
import random

def loadData(filename):
    '''
    加载数据集
    filename: 数据集路径
    返回: 数据集和标签集
    '''
    

    dataArr = []; labelArr = []    #存放数据及标签
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) / 255 for num in curLine[1:]])

        #将数字0标记为1，其余数字标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    return dataArr, labelArr

class SVM:
    '''
    支持向量机类
    '''
    def __init__(self, trainDataList, trainLabelList, sigma=10, C=200, toler=0.001):
        '''
        初始化参数
        '''
        self.trainDataMat = np.mat(trainDataList)       # 训练集数据
        self.trainLableMat = np.mat(trainLabelList).T   # 列向量形式的训练标签集
        
        self.sigma = sigma
        self.m , self.n = np.shape(self.trainDataMat)   # m: 训练样本的个数；n: 训练样本的特征数784
        self.C = C   #软间隔的惩罚参数
        self.toler = toler  #松弛变量

        self.k = self.calcKernel()
        self.b = 0 #偏置b设为0
        self.alpha = [0] * self.trainDataMat.shape[0]  # alpha的长度为训练集的数目
        self.E = [0 * self.trainLableMat[i, 0] for i in range(self.trainLableMat.shape[0])]
        self.supportVecIndex = []

    def calcKernel(self):
        '''
        计算高斯核函数
        return: 高斯核矩阵
        '''
        #初始化高斯核矩阵大小， 即一个m行m列的矩阵
        k =[[0 for i in range(self.m)] for j in range(self.m)] #k是一个二维的列表，访问下标要用k[i][j]
        
        for i in range(self.m):

            if i % 100 == 0:
                print('construct the kernel:', i, self.m)

            X = self.trainDataMat[i,:] #第i个训练数据

            for j in range(i, self.m):

                Z = self.trainDataMat[j,:]

                temp = (X - Z) * (X - Z).T
                res = np.exp(-1 * temp / (2 * self.sigma))

                #由于核矩阵的对称性，可以只算一半
                k[i][j] = res
                k[j][i] = res


        return k

    def isSatisfyKKT(self, i):
        '''
        判断第i个alpha是否满足KKT条件
        参数：i alpha的下标
        返回：True满足；False不满足
        '''
        gxi = self.calc_gxi(i)
        yi = self.trainLableMat[i]

        #李航统计机器学习 7.111到7.113三个KKT条件
        #7.111
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        #7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi >= 1):
            return True
        #7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) and (math.fabs(yi * gxi - 1) < self.toler):
            return True
        return False

    def calc_gxi(self, i):
        '''
        计算g(xi)
        '''
        gxi = 0

        #获取非0 alpha的下标，并装换成列表的形式方便后续遍历
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.trainLableMat[j] * self.k[j][i]

        gxi += self.b
        return gxi

    def calcEi(self, i):
        '''
        计算Ei
        参数i：E的下标
        '''
        gxi = self.calc_gxi(i)
        return gxi - self.trainLableMat[i]

    def getAlphaJ(self, E1, i):
        '''
        SMO方法中选择第二个变量
        参数 E1:第一个变量的E1
        参数 i:第一个变量α的下标
        '''
        E2 = 0 #初始化E2
        maxE1_E2 = -1 
        maxIndex = -1
        

        #h获取Ei非零的对应索引组成的列表，列表内容为非0 Ei 的下标i
        nonzeroE = [i for i , Ei in enumerate(self.E) if Ei != 0]

        for j in nonzeroE:
            E2_tmp = self.calcEi(j)
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)

                E2 = E2_tmp

                maxIndex = j
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.m))
            E2 = self.calcEi(maxIndex)
        return E2, maxIndex
        
    def train(self, iter = 100):

        iterStep = 0
        parameterChanged = 1

        while (iterStep < iter) and (parameterChanged > 0):
            print('iter:%d:%d'%(iterStep, iter))
            iterStep += 1
            parameterChanged = 0

            #遍历所有样本，寻找SMO中的第一个变量
            for i in range(self.m):
                if self.isSatisfyKKT(i) == False:
                    E1 = self.calcEi(i)
                    E2, j = self.getAlphaJ(E1, i)

                    y1 = self.trainLableMat[i]
                    y2 = self.trainLableMat[j]

                    #复制α作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]

                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_1 + alphaOld_2)
                    if L == H:
                        continue

                    #计算α的新值，7.106来更新α2值
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    #根据7.106更新α2,未剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)

                    #剪切α2
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H

                    #更新α1 7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    #根据7.115和7.116来计算b1 b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b

                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    #依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    #更新各类值
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)
        

    def calcSinglKernel(self, x1, x2):
        res = (x1 - x2) * (x1 - x2).T
        res = np.exp(-1 * res / (2 * self.sigma ** 2))
        return res

    def predict(self ,x):
        '''
        用7.94来预测x
        '''
        res = 0
        for i in self.supportVecIndex:
            temp = self.calcSinglKernel(self.trainDataMat[i,:], np.mat(x))

            res += self.alpha[i] * self.trainLableMat[i] * temp
        res += self.b
        return np.sign(res)

    def test(self, testDataList, testLabelList):
        '''
        return: 正确率
        '''
        errorCnt = 0

        for i in range(len(testDataList)):
            print('test:%d:%d'%(i, len(testDataList)))

            res = self.predict(testDataList[i])

            if res != testLabelList[i]:
                errorCnt += 1
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()

    #获取训练集以及标签
    print('start read transSet')
    trainDataList, trainLabelList = loadData('mnist_train.csv')

    # 获取测试集以及标签
    print('start read testset')
    testDataList, testLabelList = loadData('mnist_test.csv')

    # 初始化SVM类
    print('start init SVM')
    #取前1000个作为训练集
    svm = SVM(trainDataList[:1000],trainLabelList[:1000], 10,200, 0.001)


    #开始训练
    print('strat to train')
    svm.train()


    #开始测试
    print('start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is : %d' %(accuracy * 100), '%')

    print('time span:', time.time() - start)




