import numpy as np 
import time

def loadData(filename):

    dataArr = []
    labelArr = []

    fr = open(filename)

    for line in fr.readlines():
        curLine = line.strip().split(',')
        #此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])

        labelArr.append(int(curLine[0]))

    return dataArr, labelArr


def NaiveBayes(Py, Px_y, x):
    '''
    Py: 先验分布
    Px_y: 条件分布
    '''
    featureNum = 784
    classNum = 10

    P = [0] * classNum #所有标签的概率估计

    #计算x估计为每一类的概率
    for i in range(classNum):
        sum = 0
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]

        P[i] = sum + Py[i]

    return P.index(max(P))

def test(Py, Px_y, testDataArr, testLabelArr):
    errorCnt = 0
    for i in range(len(testDataArr)):
        predict = NaiveBayes(Py, Px_y, testLabelArr[i])

        if predict != testLabelArr[i]:
            errorCnt += 1

    return 1 - (errorCnt / len(testDataArr))

def getAllPro(trainDataArr, trainLabelArr):

    '''
    通过训练集计算先验分布和条件分布
    '''
    featureNum = 784
    classNum = 10

    Py = np.zeros((10,1))
    for i in range(classNum):
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)
    Py = np.log(Py)

    Px_y = np.zeros((classNum, featureNum, 2))

    for i in range(len(trainLabelArr)):
        label = trainLabelArr[i]
        x = trainDataArr[i]
        for j in range(featureNum):
            Px_y[label][j][x[j]] += 1

    for label in range(classNum):
        for j in range(featureNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    return Py, Px_y

if __name__ == '__main__':
    start = time.time()
    # 获取训练集
    print('start read transSet')
    trainDataArr, trainLabelArr = loadData('mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData('mnist_test.csv')

    #开始训练，学习先验概率分布和条件概率分布
    print('start to train')
    Py, Px_y = getAllPro(trainDataArr, trainLabelArr)

    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print('start to test')
    accuracy = test(Py, Px_y, testDataArr, testLabelArr)

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)




    
