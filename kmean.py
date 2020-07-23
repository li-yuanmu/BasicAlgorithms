import numpy as np
import random
import sys
import matplotlib.pyplot as plt
#手写K均值算法
# data是一个narray, data = np.array([[1,2],[3,4],[5,6],[7,8]])


class Kmean:

    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.points = self.start_point(data, k)
        print('初始点' + str(self.points))

    def main(self):
        res = []
        for i in range(self.k):
            res.append([])

        for item in self.data:
            dis_min = sys.maxsize
            index = -1
            #计算单个样本最近的中心
            for i in range(len(self.points)):
                dis = self.dis(item, self.points[i])
                if dis < dis_min:
                    dis_min = dis
                    index = i
            #根据中心形成不同的簇，res[0]为第一类......res[k-1]为第k类        
            res[index] = res[index] + [item.tolist()]
        
        new_center = []
        for item in res:
            new_center.append(self.center(item).tolist())
        print('初始点' + str(new_center))

        if (self.points == new_center).all():
            return res

        self.points = np.array(new_center)
        return self.main()

        pass

    def center(self, list):
        #计算中心点
        return np.array(list).mean(axis=0)

    def dis(self, p1, p2):
        #计算两点间距离
        dis = 0
        for i in range(len(p1)):
            dis += pow(p1[i] - p2[i], 2)
        return pow(dis, 0.5)

    def start_point(self, data, k):
        #计算初始点
        if k <= 0 or k > data.shape[0]:
            raise Exception('簇数设置有误！！！')

        #求随机点的下标
        indexes = random.sample(np.arange(0, data.shape[0], 1).tolist(),k)
        points = []
        for i in indexes:
            points.append(data[i].tolist())
        return np.array(points)

data = np.array([[1,2],[3,4],[8,9],[12,14]])

cluster = Kmean(data, 2)

result = cluster.main()
print(result)

for item in result:
    plt.scatter([x[0] for x in item], [x[1] for x in item])

plt.show()