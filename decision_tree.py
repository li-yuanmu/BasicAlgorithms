import numpy as np 
import time

def loadData(fileName):
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')

        dataArr.append([int(int(num) > 128) for num in curLine[1:]])

        labelArr.append(int(curLine[0]))

    return dataArr, labelArr

def majorClass(labelArr):
    '''
    找出样本个数最多的类别,返回的是数目最多的那个类别
    '''
    classDict = {}
    for i in range(len(labelArr)):
        if labelArr[i] in classDict.keys():
            classDict[labelArr[i]] += 1
        else:
            classDict[labelArr[i]] = 1

    classSort = sorted(classDict.items(), key=lambda x:x[1], reverse=True)
    return classSort[0][0]

def calc_HD(trainLabelArr):
    '''
    计算数据集D的经验熵
    '''
    #初始化为0
    H_D = 0
    trainLabelSet = set([label for label in trainLabelArr])

    for i in trainLabelSet:
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size

        H_D += -1 * p * np.log2(p)
    return H_D


def calc_HDA(trainDataArr_DevFeature, trainLabelArr):
    '''
    计算条件熵
    '''
    H_D_A = 0
    trainDataSet = set([label for label in trainDataArr_DevFeature])

    for i in trainDataSet:
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size * calc_HD(trainLabelArr[trainDataArr_DevFeature == i])

    return H_D_A

def calcBestFeature(trainDataList, trianLabelList):
    '''
    计算信息增益最大的特征
    '''
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trianLabelList)

    featureNum = trainDataArr.shape[1]

    maxG_D_A = -1
    maxFeature = -1
    H_D = calc_HD(trainLabelArr)

    for feature in range(featureNum):

        trainDataArr_DevideByFeature = np.array(trainDataArr[:,feature].flat)

        G_D_A = H_D - calc_HDA(trainDataArr_DevideByFeature,trainLabelArr)

        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            maxFeature = feature
    return maxFeature, maxG_D_A

def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    '''

    '''
    retDataArr = []
    retLabelArr = []

    for i in range(len(trainDataArr)):
        if trainDataArr[i][A] == a:
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])

            retLabelArr.append(trainLabelArr[i])
    return retDataArr, retLabelArr


def creatTree(*dataSet):
    '''
    递归创建决策树
    return: 新的子节点或者该叶子节点的值
    '''
    Epsilon = 0.1

    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]

    print('start a node', len(trainDataList[0]), len(trainLabelList))

    classDict = {i for i in trainLabelList}

    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)

    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)

    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)

    treeDict = {Ag:{}}

    treeDict[Ag][0] = creatTree(getSubDataArr(trainDataList, trainDataList, Ag, 0))
    treeDict[Ag][1] = creatTree(getSubDataArr(trainDataList, trainDataList, Ag, 1))

    return treeDict

def predict(testDataList, tree):
    while True:
        (key, value), = tree.items()

        if type(tree[key]).__name__ == 'dict':
            #需要不停的删掉用完的特征
            dataVal = testDataList[key]
            del testDataList[key]
            #将tree更新为其子节点的字典
            tree = value[dataVal]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value

def model_test(testDataList, testLabelList, tree):
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time

    trainDataList, trainLabelList = loadData('mnist_train.csv')

    testDataList, testLabelList = loadData('mnist_test.csv')

    print('start create tree')
    tree = creatTree((trainDataList, trainLabelList))
    print('tree is :', tree)

    print('start test')
    accur = model_test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    end = time.time

    print('time span:' ,end - start)