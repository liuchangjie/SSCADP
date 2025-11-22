import numpy as np
import pandas as pd
import copy
import sys

class Concept_drift():
    def calcu_r_single_cluster(self, members, center): #计算单个簇的半径
        data_array = copy.deepcopy(members)
        center_array = copy.deepcopy(center)
        total_members = len(data_array)  # 一个簇中的样本总数
        r_sum = 0
        for i in range(len(data_array)): #计算各个样本到中心点的欧式距离
            dis = np.sqrt(np.sum(np.square(data_array[i]-center_array))) #单个样本到中心的距离
            r_sum = r_sum + dis

        return r_sum / total_members

    def calcu_r(self, cluster_info, center_list):
        r_sum = 0
        num_cluster = len(center_list)
        for key,value in center_list.items():
            center = value
            members = cluster_info[key]
            r = self.calcu_r_single_cluster(members, center)
            r_sum = r_sum + r

        return r_sum / num_cluster

    def calcu_dist(self, center_hist, center_new):
        min_dist_list = []
        for value_new in center_new.values():
            min_dist = sys.float_info.max
            for value_hist in center_hist.values():
                dist = np.sqrt(np.sum(np.square(value_new-value_hist)))
                if dist < min_dist:
                    min_dist = dist
            min_dist_list.append(min_dist)
        distance = sum(min_dist_list) / len(center_new)

        return distance

    def calcu_dist_weight(self, center_hist, data_index_hist, center_new, data_index_new):
        # print('center_hist:',center_hist)
        # print('center_new:', center_new)
        data_dict_hist = copy.deepcopy(data_index_hist)
        data_dict_new = copy.deepcopy(data_index_new)
        for key, value in data_dict_hist.items():
            new_value = len(value)     #获得簇大小
            data_dict_hist[key] = new_value
        # print('data_dict_hist:',data_dict_hist)
        sorted_dict_hist = sorted(data_dict_hist.items(), key=lambda x : x[1], reverse=True) #按值对字典进行降序排列
        # print('sorted_dict_hist',sorted_dict_hist)

        for key, value in data_dict_new.items():
            new_value_ = len(value)
            data_dict_new[key] = new_value_
        # print('data_dict_new:', data_dict_new)
        sorted_dict_new = sorted(data_dict_new.items(), key=lambda x : x[1], reverse=True)
        # print('sorted_dict_new:', sorted_dict_new)

        len_1 = len(sorted_dict_hist)
        len_2 = len(sorted_dict_new)
        min_len = min(len_1, len_2)
        sorted_dict_hist = sorted_dict_hist[0:min_len]
        sorted_dict_new = sorted_dict_new[0:min_len]
        num_hist = sum([example[1] for example in sorted_dict_hist])
        num_new = sum([example[1] for example in sorted_dict_new])
        num_data = num_hist + num_new
        dist_list = []
        for i in range(len(sorted_dict_new)):
            idx_hist = sorted_dict_hist[i][0]
            idx_new = sorted_dict_new[i][0]
            w = (sorted_dict_new[i][1]+sorted_dict_hist[i][1]) / num_data
            dist = w * (np.sqrt(np.sum(np.square(center_new[idx_new]-center_hist[idx_hist]))))
            dist_list.append(dist)
        # print('dist_list:',dist_list)
        # print('--------------------------')
        return sum(dist_list)


    def judge_drift(self, dist, r_hist, r_new):
        flag = False
        if dist >= (r_hist + r_new):
            flag = True
        return flag





