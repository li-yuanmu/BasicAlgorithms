import numpy as np 
import time
'''
二分类问题的准确度为80%
'''

def loadData(fileName):

    print('start read file')

    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')

        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)


        dataArr.append([int(num) for num in curLine[1:]])
        

    return dataArr, labelArr


def perceptron(dataArr, labelArr, iter=50):
    '''
    返回训练好的w和b
    '''
    print('start to trans')

    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    m, n = np.shape(dataMat)

    w = np.zeros((1, n))
    b = 0
    h = 0.0001

    for k in range(iter):
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]

            #判断是否误分
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi

        print('Round %d:%d traing' %(k,iter))
    return w, b

def test(dataArr, labelArr, w, b):
    print('start to test')

    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    m, n = np.shape(dataMat)

    errorCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]

        res = -1 * yi * (w * xi.T + b )

        if res >= 0: errorCnt += 1

    accru = 1 - (errorCnt / m)

    return accru

if __name__ == '__main__':

    start = time.time()

    trainData, trainLabel = loadData('mnist_train.csv')
    testData, testLabel = loadData('mnist_test.csv')

    w, b = perceptron(trainData, trainLabel, iter=30)
    accru = test(testData, testLabel, w, b)
    end = time.time()

    print('accuracy rate is:', accru)
    print('time span:', end - start)


