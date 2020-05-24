import numpy as np 
import time

'''
25-NN: 欧氏距离的准确率为97%
        曼哈顿距离的准确度14%
'''

def loadData(fileName):

    print('start read file')

    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))

    return dataArr, labelArr


def calcDist(x1, x2):
    '''
    计算两个向量之间的距离
    '''
    dis = np.sqrt(np.sum(np.square(x1 - x2)))# 欧氏距离
    #dis = np.sum(x1 - x2)#曼哈顿距离
    return dis

def getClosest(trainDataMat, trainLabelMat, x, K):
    '''
    预测样本x的标签
    找到与样本x最近的K个点，并查看他们的标签
    '''

    #存放x与训练集中每个样本的距离的列表
    distList = [0] * len(trainDataMat)

    #遍历所有训练样本，计算与x的距离
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        temp = calcDist(x1, x)
        distList[i] = temp

    topKList = np.argsort(np.array(distList))[:K]  # 最近的K个样本对应的下标

    labelList = [0] * 10

    #遍历K个索引
    for i in topKList:
        labelList[int(trainLabelMat[i])] += 1
    return labelList.index(max(labelList))

def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, K):

    print('start test')

    trainDataMat = np.mat(trainDataArr)
    trainLabelMat = np.mat(trainLabelArr).T

    testDataMat = np.mat(testDataArr)
    testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0

    for i in range(200):

        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, K)
        if y != testLabelMat[i]:
            errorCnt += 1

    return 1 - (errorCnt / 200)


if __name__ == "__main__":
    start = time.time()

    trainDataArr, trainLabelArr = loadData('mnist_train.csv')

    testDataArr, testLabelArr = loadData('mnist_test.csv')

    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)

    print('accur is:%d'%(accur * 100), '%')


    end = time.time()

    print('time span:', end - start)