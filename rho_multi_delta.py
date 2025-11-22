"""
利用density peak cluster算法计算落入叶子节点样本的rho*delta值，用于后续使用change_point检测跳跃点，自动确定簇的数量
12/13 加入聚类过程
"""
import numpy as np
import pandas as pd
import sys
import math
from collections import defaultdict
from change_point import Change_point
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Rho_multi_Delta():
    def __init__(self, data_array):
        self.data_array = copy.deepcopy(data_array)

    def calcu_distance(self):
        distance_reslut = []
        for i in range(len(self.data_array)):
            for j in range(i, len(self.data_array)):
                dis_temp = []
                dis = np.sqrt(np.sum(np.square(self.data_array[i]-self.data_array[j])))
                dis_temp.extend([i+1, j+1, dis])
                distance_reslut.append(dis_temp)
        distance_reslut = np.array(distance_reslut)
        # print(distance_reslut)  distance_reslut=[[1,1,0],[1,2,10.3],...[999,1000,6.11],[1000,1000,0]
        return distance_reslut

    def load_data(self, distance_reslut):
        distances = {}   #用字典保存样本之间的距离文件
        min_dis, max_dis = sys.float_info.max, 0.0  #min_dis记录某次计算中样本之间的最小值，max_dis记录样本之间的最大值
        max_id = 0

        for data in distance_reslut:
            x1 = int(data[0])
            x2 = int(data[1])
            dis = float(data[2])
            max_id = max(max_id, x1, x2)
            min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
            distances[(x1, x2)] = dis
            distances[(x2, x1)] = dis
        for i in range(max_id+1):
            distances[(i, i)] = 0.0

        return distances, max_dis, min_dis, max_id

    def select_dc(self, max_id, max_dis, min_dis, distances, auto = False):
        if auto:
            return self.autoselect_dc(max_id, max_dis, min_dis, distances)
        percent = 2.0
        position = int(max_id * (max_id + 1) / 2 * percent / 100)
        dc = sorted(distances.values())[position * 2 + max_id]
        # logger.info("PROGRESS: dc - " + str(dc))
        return dc

    def autoselect_dc(self, max_id, max_dis, min_dis, distances):
        dc = (max_dis + min_dis) / 2

        while True:
            nneighs = sum([1 for v in distances.values() if v < dc]) / max_id ** 2
            if nneighs >= 0.01 and nneighs <= 0.02:
                break
            # binary search
            if nneighs < 0.01:
                min_dis = dc
            else:
                max_dis = dc
            dc = (max_dis + min_dis) / 2
            if max_dis - min_dis < 0.0001:
                break
        # logger.info("PROGRESS: dc - " + str(dc))
        return dc

    #计算局部密度
    def local_density(self, max_id, distances, dc, guass=True, cutoff=False):
        '''
        Compute all points' local density

        Args:
            max_id    : max continues id
            distances : distance dict
            gauss     : use guass func or not(can't use together with cutoff)
            cutoff    : use cutoff func or not(can't use together with guass)

        Returns:
            local density vector that index is the point index that start from 1
        '''
        assert guass and cutoff == False and guass or cutoff == True
        # logger.info("PROGRESS: compute local density")
        if dc == 0: dc = 0.00003
        guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = guass and guass_func or cutoff_func
        rho = [-1] + [0] * max_id  # 多加一个-1使得index从1开始
        for i in range(1, max_id):
            for j in range(i + 1, max_id + 1):
                rho[i] += func(distances[(i, j)], dc)
                rho[j] += func(distances[(i, j)], dc)
            # if i % (max_id / 10) == 0:
            #     logger.info("PROGRESS: at index #%i" % (i))
        return np.array(rho, np.float32)

    # 计算每个点到比它密度更高的点中的最小距离 即delta
    def min_distance(self, max_id, max_dis, distances, rho):
        '''
        Compute all points' min distance to the higher local density point(which is the nearest neighbor)

        Args:
            max_id    : max continues id
            max_dis   : max distance for all points
            distances : distance dict
            rho       : local density vector that index is the point index that start from 1

        Returns:
            min_distance vector, nearest neighbor vector
        '''
        # logger.info("PROGRESS: compute min distance to nearest higher density neigh")
        sort_rho_idx = np.argsort(-rho)  # 得到一个按rho值从大到小排序对应的索引 sort_rho_idx为索引
        # delta=[0.0,max_dis,...,max_dis,...] 长度和rho相等
        delta, nneigh = [0.0] + [float(max_dis)] * (len(rho) - 1), [0] * len(rho)  # len(rho)=1000
        delta[sort_rho_idx[0]] = -1.  # 将rho值最大的点的delta值设置为-1
        for i in range(1, max_id):  # 除密度最大的那个索引开始遍历
            for j in range(0, i):  # 在密度比i高的索引中寻找
                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]  # old_i, old_j是样本点的索引
                if distances[(old_i, old_j)] < delta[old_i]:
                    delta[old_i] = distances[(old_i, old_j)]  # 得到每个点的delta值
                    nneigh[old_i] = old_j  # 存储每个点的密度高于它且最近的点的索引
            # if i % (max_id / 10) == 0:
            #     logger.info("PROGRESS: at index #%i" % (i))
        delta[sort_rho_idx[0]] = max(delta)  # 得到局部密度最大的那个点的delta值即max_dis
        return np.array(delta, np.float32), np.array(nneigh, np.float32)  # 以ndarray格式返回

    def norma_rho_delta(self, rho, delta): #归一化rho和delta的值
        rho_ = rho[1:] #除去index=0为-1的元素 数据类型还是ndarray
        Max_r = np.max(rho_)
        Min_r = np.min(rho_)
        range_r = Max_r - Min_r
        norma_rho = (rho_ - Min_r) / range_r
        list_rho = list(norma_rho)
        list_rho.insert(0,-1)
        new_rho = np.array(list_rho)

        delta_ = delta[1:]
        Max_d = np.max(delta_)
        Min_d = np.min(delta_)
        range_d = Max_d - Min_d
        norma_delta = (delta_ - Min_d) / range_d
        list_delta = list(norma_delta)
        list_delta.insert(0,0)
        new_delta = np.array(list_delta)
        return new_rho, new_delta

    def cal_rhodelta(self, nor_rho, nor_delta):  # 计算归一化后的rho*delta
        vec_rhodelta = []
        y = nor_rho * nor_delta
        y_list = list(y)       #y_list无序
        y_list_rev = sorted(y_list, reverse=True)
        vec_rhodelta.extend(y_list_rev) #若用append 则为[[1,0.987,0.876,...]]
        return vec_rhodelta, y_list

    def cluster(self):
        distance_result = self.calcu_distance()
        # print('distance_result:',distance_result)

        # 得到距离的字典形式、样本之间最大的距离、最小的距离和样本数量
        distances, max_dis, min_dis, max_id = self.load_data(distance_result)

        #dc
        dc = self.select_dc(max_id, max_dis, min_dis, distances, auto=False)
        # print('dc:',dc)

        #rho
        rho = self.local_density(max_id, distances, dc)
        # print('rho:',rho,len(rho))

        #delta
        delta, nneigh = self.min_distance(max_id, max_dis, distances, rho)

        #normalized rho and delta
        nor_rho, nor_delta = self.norma_rho_delta(rho, delta)

        #normalized rho*delta
        rho_multi_delta,unord_rho_m_delta = self.cal_rhodelta(nor_rho, nor_delta) #有序和无序的rho*delta
        unord_rho_m_delta = np.array(unord_rho_m_delta)
        sort_rho_m_delta_index = np.argsort(-unord_rho_m_delta) #rho*delta从大到小排序对应的样本索引

        return rho_multi_delta, sort_rho_m_delta_index, rho, nneigh


    #先只处理连续属性 聚类过程
    def cluster_result(self, num_cluster, sort_rho_m_delta_index, rho, nneigh, data ):#依据自动确定的簇数量，根据密度大小选出聚类中心，再将样本分配至各个聚类中心
        '''
        :param num_cluster: 聚簇的个数
        :param rho:         每个样本的密度
        :param sort_rho_m_delta_index: rho*delta值从大到小的样本索引
        :param data         样本集合
        :return:
        '''
        cluster = defaultdict(list)  #初始化一个有默认值的空字典，如{'f':[]} 记录聚类信息
        data_index_in_cluster = defaultdict(list)  #用于保存每个聚簇中所有样本的索引
        cencetr = {}                  #保存聚类中心点的样本
        cluster_temp = {}             #临时保存每个数据的聚类类别结果
        ordrho = np.argsort(-rho)
        # print('num_cluster:',num_cluster)
        cluster_index = sort_rho_m_delta_index[0:num_cluster] #选定的簇中心的索引 [17,15,70,25]四个rho*delta最大的值在data中的索引
        # print('cluster_index:',cluster_index)
        for index in cluster_index:
            cencetr[index] = data[index-1]  #{17:[2.3,5.6,7.8],15:[8.6.4.3,9.1],}
        # print('cencetr:',cencetr)

        cluster_index_rho = []
        for id in cluster_index:
            cluster_index_rho.append(rho[id])
        # print('cluster_index_rho:', cluster_index_rho)


        for idx in cluster_index:
            cluster_temp[idx] = idx

        for idx, (ldensity, nneigh_item) in enumerate(zip(rho, nneigh)):
            if idx == 0 or idx in cluster_temp: continue
            else:
                cluster_temp[idx] = -1
        # print('nneigh:', nneigh)
        # assignation 得到的cluster = {17:17,15:15,70:70,25:25,0:17,1:17,2:70,.....199:25}
        last_nneigh_idx = None
        for j in range(len(ordrho)-1): #遍历每一个索引
            idx = ordrho[j]
            if idx == 0: continue  # index=0
            if cluster_temp[idx] == -1: #如果当前样本点是非聚类中心
                # print('nneigh[idx]:',nneigh[idx])
                if nneigh[idx] != 0:
                    cluster_temp[idx] = cluster_temp[nneigh[idx]] #当前点的标签和高于当前点密度的最近的点的标签一致
                    last_nneigh_idx = nneigh[idx]
                else:
                    if last_nneigh_idx == None:
                        cluster_temp[idx] = cluster_temp[nneigh[idx-1]]

                    else:cluster_temp[idx] = cluster_temp[last_nneigh_idx]


        #以字典形式得到簇类信息
        for key, value in cluster_temp.items(): #key为样本的索引，value为样本属于哪个高密度点的索引
            cluster[value].append(data[key-1])  #note!!因为之前方便计算rho,delta加了一个多的索引，加入对应的数据需要减1
            data_index_in_cluster[value].append(key-1) #用于保存每个聚簇中所有样本的索引，方便获得标签进行半监督学习

        # print('cluster:',cluster)
        # print('data_index:',data_index_in_cluster)
        # print('-----beginning')
        # for values in data_index_in_cluster.values():
        #     print('每个簇中样本的数量',len(values))
        # print('----ending----')

        # color = ['red','black','blue','green','yellow','gray','purple','orange']
        # i = 0
        # # fig = plt.figure()
        # # ax = Axes3D(fig)
        # for values in cluster.values():
        #     datas = np.array(values)
        #     tsne = TSNE()
        #     tsne.fit_transform(values)
        #     tsne = pd.DataFrame(tsne.embedding_)
        #     tsne= tsne.values
        #     print('tsne:',tsne)
        #     x = tsne[:,0]
        #     y = tsne[:,1]
        #     plt.scatter(x, y, c=color[i], s=8)
        #     i += 1
        #     # x = datas[:,0]
        #     # y = datas[:,1]
        #     # # z = datas[:,2]
        #     #
        #     # # ax.scatter(x, y, z, c = color[i], s = 10)
        #     # plt.scatter(x, y, c=color[i], s=8)
        #     # i += 1
        #
        # plt.show()

        return cluster, cencetr, data_index_in_cluster


def write_rhodelta_to_csv(vec_rhodelta):
    new_vec = np.array(vec_rhodelta)
    new_vec = pd.DataFrame(new_vec, dtype='float32')
    new_vec.to_csv('D:/datasets/Sea/norma/Sea-rd-nor-10-new.csv', header=0, index=0)



if __name__ == '__main__':
    df = pd.read_csv('D:/datasets/Sea/Sea-10.csv', header=None, sep=',') #测试样本1000个
    df = df.values
    data = df[:,:-1] #去掉标签信息
    rhodelta = Rho_multi_Delta(data) #初始化一个Rho*Delta对象

    rho_multi_delta,sort_rho_m_delta_index, rho, nneigh = rhodelta.cluster() #返回归一化的rho*delta
    rhodelta.cluster_result(4, sort_rho_m_delta_index, rho, nneigh, data)

    # write_rhodelta_to_csv(vec_rhodelta)

