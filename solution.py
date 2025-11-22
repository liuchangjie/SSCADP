# Very Fast Decision Tree i.e. Hoeffding Tree, described in
# "Mining High-Speed Data Streams" (Domingos & Hulten, 2000)
#
# this program contains 2 classes: Vfdt, VfdtNode
# changed to CART: gini index
#
# Jamie
# 02/06/2018
# ver 0.03

# from sklearn.semi_supervised import LabelPropagation

'''
solution_3:when labeling the unlabel data,using the density peak cluster algorithm.
In each cluster,label propagation is used to label the unlabel data.

When detecting the drift,determined the number of cluster automatically.
Based on the difference of hist cluster set and new cluster set to detect the drift.
Add the weighting method based on the cluster's density

'''
import numpy as np
import pandas as pd
import time
from itertools import combinations  # itertools 迭代器，combinations实现排列组合
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import copy
from collections import Counter
import random
from rho_multi_delta import Rho_multi_Delta
from change_point import Change_point
# from SUN.original_drift_detection import Drift_detection
from concept_drift_detect import Concept_drift
from sklearn.semi_supervised import LabelPropagation #标签传播
import warnings
warnings.filterwarnings("ignore")

# VFDT node class
class VfdtNode:
    def __init__(self, possible_split_features):  # __init__构造方法，初始化
        """
        nijk: statistics of feature i, value j, class k 统计量字典形式
        :list possible_split_features: features 可能的分裂属性 列表形式
        """
        self.parent = None            # parent 每一个VfdtNode实例化的对象自带这些parent,left_child...等属性
        self.brother = None
        self.left_child = None        # left_child
        self.right_child = None       # right_child
        self.split_feature = None     # 选择的划分属性
        self.split_value = None       # both continuous and discrete value离散青绿、乌黑；连续含糖率等
        self.new_examples_seen = 0    # 节点新看到的样本
        self.total_examples_seen = 0  # 节点总共看到的样本
        self.class_frequency = {}     # 定义一个字典存放节点中的类及其对应的数量，key:class; value:count
        self.nijk = {f: {} for f in possible_split_features} # nijk={'a':{},'b':{},'c':{}}
        self.possible_split_features = possible_split_features
        # self.k_value = k_value      # kmodes中的k值
        self.data_array = []          # 用于保存当前叶子节点累积的数据，用于kmeans聚类打标记
        self.part_label = []          # 用于保存当前叶子节点累积的数据的标记
        self.cluster_hist = None # 用于保存叶子节点的历史概念簇集的聚簇信息，用于概念漂移检测
        self.cluster_new = None  # 新概念簇集聚簇信息
        self.center_hist = None       # 历史概念簇集的聚簇中心
        self.center_new = None        # 新概念簇集的聚簇中心
        self.data_index_hist = None   #与cluster对应 每个样本点替换成了在当前数据中的索引
        self.data_index_new = None
        # self.intervals_hist = None    # 保存每个历史聚簇中心离散化后连续属性对应的区间信息
        # self.intervals_new = None     # 保存每个新聚簇中心离散化后连续属性对应的区间信息

    def add_children(self, split_feature, split_value, left, right):  # add_children方法
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left         #left，right都是一个VfdtNode对象
        self.right_child = right
        left.parent = self    # 左叶子节点的父节点指向当前节点
        right.parent = self   # 右叶子节点的父节点指向当前节点
        left.brother = right  # 定义左孩子的兄弟节点
        right.brother = left  # 定义右孩子的兄弟节点

        self.nijk.clear()  # reset stats 重置为一个空的字典，节点分裂之后，原有的统计信息不必保存
        if isinstance(split_value, list):  # 如果第一个参数是第二个参数的实例对象，则返回true
            left_value = split_value[0]    # 如果split_value是一个列表
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting
            # 离散分裂值列表的长度为1时，停止分裂
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):  # 判断是否为叶子节点
        return self.left_child is None and self.right_child is None #二者同时成立返回True

    # recursively trace down the tree to distribute dataset examples to corresponding leaves
    #递归地将样本落入至叶子节点
    def sort_example_train(self, x, y):
        if self.is_leaf():             # 如果是叶子节点
            self.data_array.append(x)  # 保存落入当前叶子节点中的数据
            self.part_label.append(y)  # 保存落入叶子结点数据的标签（半监督）未标记样本的标签为-1
            self.new_examples_seen += 1
            return self
        else:  # 如果(x,y)落入非叶子节点，则递归按属性落入叶子节点
               # index定位分裂属性列表中的分裂属性的位置
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]   #x为样本，具有多维属性
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value 离散属性
                if value in split_value[0]:    # isinstance()如果第一个参数是第二个的实例对象返回true
                    return self.left_child.sort_example_train(x, y)  # 递归调用sort_example()函数
                else:
                    return self.right_child.sort_example_train(x, y)
            else:  # continuous value 连续属性
                if value <= split_value:
                    return self.left_child.sort_example_train(x, y)
                else:
                    return self.right_child.sort_example_train(x, y)

    def sort_example(self, x): #分类测试阶段，不需要用到标签信息
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:  # class_frequency是一个字典，max遍历字典取最大，.get获取最大值对应的键值（类）
              #不加key这个函数的话，默认遍历的是字典的key，最后输出最大的键
            prediction = max(self.class_frequency, key=self.class_frequency.get)

        except ValueError:
            # if self.class_frequency dict is empty, go back to parent 如果字典为空，返回父节点
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats（叶子节点的统计信息） in order to calculate G()
    def update_stats(self, x, y):
        feats = self.possible_split_features #possible_split_features类型是list

        # 由于做的是当一个叶子节点中的样本都获得标记后批量处理统计信息，所以不采用原文的一个一个样本的更新方法，
        nijk = self.nijk
        if not bool(nijk):#如果nijk为空 liu
            nijk = {f: {} for f in feats}

        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]  #找到样本对应属性的取值
            if value not in nijk[i]:
                nijk[i][value] = {y: 1} #y代表类别 不在原有的字典中，新增一个记录
            else:
                try:
                    nijk[i][value][y] += 1
                except KeyError:  # keyError 字典中查找一个不存在的关键字
                    nijk[i][value][y] = 1

        class_frequency = self.class_frequency  # 一个节点中各个类别及其对应数量的统计
        self.total_examples_seen += 1   #记录叶子节点总共落入的样本数，用于后续计算epsilon
        # self.new_examples_seen += 1   #前面sort_example()时叶子节点记录过，记录叶子节点新落入的样本数 11/10

        #记录当前叶子节点的类别分布
        try:
            class_frequency[y] += 1
        except KeyError:  # keyError 字典中查找一个不存在的关键字，
            class_frequency[y] = 1 #遇到新类

    def check_not_splitting(self):
        # compute gini index for not splitting 计算没有划分时的基尼指数
        X0 = 1
        class_frequency = self.class_frequency
        # print(class_frequency) #{'no':179 , 'yes':21}
        # print(len(class_frequency)) # len=2
        # print(class_frequency.values()) #dict_values([179,21]) 字典中的.values()用于返回字典中的所有值
        n = sum(class_frequency.values())  # n=200对应分割阈值
        # print(class_frequency.items()) dict_items([('no', 179), ('yes', 21)])返回字典中所有的项
        for j, k in class_frequency.items():
            X0 -= (k / n) ** 2
        return X0


    def attempt_split(self, delta, nmin, tau): # use Hoeffding tree model to test node split, return the split feature
        if self.new_examples_seen < nmin:  #未达到最小检测分割的数量
            return None                    #return None直接退出函数

        class_frequency = self.class_frequency
        if len(class_frequency) == 1:  # 字典中的项的个数为1，如果只有一类不划分
            return None

        self.new_examples_seen = 0  #满足分裂测试时，叶子节点新观测到的样本清0，若未分裂，则用于累积下次分裂所需新样本
        self.data_array=[]          #nmin个样本打上标记并记录统计信息后，开始尝试分裂，不论是否分裂，用于保存nmin中数据的数组
        self.part_label=[]          #将清空，用来保存下次到来的新的样本和标签，达到nmin后再尝试分裂 11/10
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        # print(self.possible_split_features) #possible_s_f=['age','job',...'poutcome']对应所有属性
        for feature in self.possible_split_features:
            if feature is not None:
                # print(nijk)#{'age':{44:{'no':6},...35{'yes':8}...},...'job':{'unem':{'no':9}...}}}
                njk = nijk[feature]
                # print(njk) #属性i对应的取值的 类别的数量
                # {'married': {'no': 117, 'yes': 13}, 'single': {'no': 42, 'yes': 6}, 'divorced': {'no': 18, 'yes': 4}}
                gini, value = self.gini(njk, class_frequency) #计算在某个属性上最优的gini指数及对应的取值
                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                    # print(split_value)#连续属性 eg:60.5 离散属性 [['married'], ['divorced', 'single']]m在一组，d和s在一组
                elif min < gini < second_min:
                    second_min = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            if second_min - min > epsilon:
                # print('1 node split')
                return [Xa, split_value]
            elif second_min - min < epsilon < tau:  # ΔH<e<t
                # print('2 node split')
                return [Xa, split_value]
            else:
                return None   #该方法要么返回最优划分属性及其属性取值，要么不做任何处理
        return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen #落入叶子节点的总数
        R = np.log2(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1 / delta) / (2 * n))  # 求epsilon的值

    def gini(self, njk, class_frequency):
        # gini(D) = 1 - Sum(pi^2) 数据集D的纯度
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2) 属性f的基尼指数，选择使得划分后gini最小的属性作为最优属性

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        # print(njk)#{'married': {'no': 111, 'yes': 8}, 'single': {'yes': 8, 'no': 45}, 'divorced': {'no': 26, 'yes': 2}}
        feature_values = list(njk.keys())  # list() is essential
        # print(feature_values) #['married', 'single', 'divorced'] 返回字典中的键,用列表记录属性的每个取值
        if not isinstance(feature_values[0], str):  # numeric  feature values 连续属性的取值
            sort = np.array(sorted(feature_values))  # 连续属性离散化 排好序的连续属性值
            # print(sort)#sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
            split = (sort[0:-1] + sort[1:]) / 2  # vectorized computation, like in R
            # print(split)                      #两个错开的数组相加再除以2,得到划分点集合
            # print(sort[0:-1],sort[1:])
            D1_class_frequency = {j: 0 for j in class_frequency.keys()}  # 获得字典的键值
            # print(D1_class_frequency)#{'no': 0, 'yes': 0}
            for index in range(len(split)):
                nk = njk[sort[index]]
                # print(nk)#一层层剥开的感觉，属性的某个取值对应的类别数量，比如属性age 25岁对应的类别数量
                # nk={'no':71,'yes':3}
                for j in nk:
                    D1_class_frequency[j] += nk[j]
                    # D1_class_frequency=D1_class_frequency + nk[j]
                    # print(D1_class_frequency)#{'no':14,'yes':2}
                    # print(D1_class_frequency.values()) #dict_values([14,2])
                D1 = sum(D1_class_frequency.values())  # sum=16
                D2 = D - D1  # D = self.total_examples_seen=nmin = 200
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    # print(class_frequency)
                    # print(D1_class_frequency)
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                        # print(D2_class_frequency)
                        # print('\n')
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v / D1) ** 2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v / D2) ** 2
                g = g_d1 * D1 / D + g_d2 * D2 / D
                if g < m1:  # m1=1
                    m1 = g
                    Xa_value = split[index]  # 连续属性离散化时的划分点集合
                # elif m1 < g < m2:
                # m2 = g
            return [m1, Xa_value]  # 返回基尼指数和分割点

        else:  # discrete feature_values
            length = len(njk)
            # print(njk)#{'married': {'no': 111, 'yes': 8}, 'single': {'yes': 8, 'no': 45}, 'divorced': {'no': 26, 'yes': 2}}
            if length > 10:  # too many discrete feature values, estimate 离散属性的取值超过10个
                for j, k in njk.items():

                    D1 = sum(k.values())  # {'no': 111, 'yes': 8} 类别的总数
                    D2 = D - D1  # D=200
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():  # class_frequency={'no':183,'yes':17}
                        # print(class_frequency)
                        if key in k:  # n=200
                            # print(key) #key为键值
                            # print(k) #k为类别字典 如{'no': 40, 'yes': 3}
                            D2_class_frequency[key] = value - k[key]
                            # {'no':183-40,'yes':17-3}
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v / D1) ** 2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                    # elif m1 < g < m2:
                    # m2 = g
                # print(feature_values)
                # print(Xa_value)
                right = list(np.setdiff1d(feature_values, Xa_value))  #非最优属性集，np.setdiff1d找出两个数组的集合差
                # print(right) #job这个离散属性的取值超过10个

            else:  # fewer discrete feature values, get combinations 少于10个离散取值采用组合方式
                comb = self.select_combinations(feature_values)
                # print(type(comb)) <class 'list'>
                for i in comb:
                    left = list(i)
                    # print(left)
                    D1_class_frequency = {key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {key: 0 for key in class_frequency.keys()}
                    # print(D1_class_frequency,D2_class_frequency) #{'no': 0, 'yes': 0} {'no': 0, 'yes': 0}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            # print(class_frequency)#{'no': 189, 'yes': 11}
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v / D1) ** 2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g  # m1的初始值为1
                        Xa_value = left
                    # elif m1 < g < m2:
                    # m2 = g
                # print(left) #['divorced']或者['primary', 'unknown']
                # print(len(comb))
                right = list(np.setdiff1d(feature_values, Xa_value))  #非最优属性集，np.setdiff1d找出两个数组的集合差
            return [m1, [Xa_value, right]]  # 返回基尼指数

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e / 2)
            for i in range(1, end + 1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb) / 2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e - 1) / 2)
            for i in range(1, end + 1):
                combination.extend(combinations(feature_values, i))
        # print(combination)#[('married',), ('divorced',), ('single',)]
        return combination


# very fast decision tree class, i.e. hoeffding tree #i.e.= in other words
class Vfdt:
    def __init__(self, features, delta=0.0000001, nmin=200, tau=0.05):  # 类的__init__构造方法
        """
        :features: list of dataset features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        # self.k_value = k_value
        # self.root = VfdtNode(features,k_value)  # 对象root为一个实例化的VfdtNode 11/10
        self.root = VfdtNode(features)        # 对象root为一个实例化的VfdtNode
        self.n_examples_processed = 0

    # update the tree by adding training example
    def update(self, x, y):
        self.n_examples_processed += 1             #VFDT模型总共处理的过的样本数量
        node = self.root.sort_example_train(x,y)   #定位到当前(x,y)样本落入的叶子节点
        if node.new_examples_seen == self.nmin:    #如果累积到当前叶子节点的数量满足最小检测分裂的数量
            pre_x_train = np.array(node.data_array)
            pre_y_train = np.array(node.part_label)

            x_train, y_train, cluster, center, data_index = self.semi_method(pre_x_train, pre_y_train) #给未标记样本打上标记
            for x_1, y_1 in zip(x_train, y_train):
                node.update_stats(x_1, y_1)        #更新新样本落入的叶节点的统计信息

            result = node.attempt_split(self.delta, self.nmin, self.tau) #尝试分裂叶节点
            # print(result) #[['duration'], 348.0]

            if result is None: #当前叶节点不分裂
                if node.cluster_hist is None: #历史概念簇集为空，第一次进行检测采用密度聚类方法，记录节点的历史概念簇集合信息
                    node.center_hist = center
                    node.cluster_hist = cluster
                    node.data_index_hist = data_index

                else: #满足当前叶节点不分裂且历史概念簇集不为空则判断是否发生概念漂移
                    brother_node = node.brother  #当前叶子节点的兄弟节点
                    self.detect_concept_drift(node, brother_node, center, cluster, data_index)

            if result is not None:   # result不为空，当前叶子节点继续分裂
                feature = result[0]
                value = result[1]
                self.node_split(node, feature, value)


    #judge the dist and (r_hist,r_new)
    def judge(self, center_hist, cluster_hist, data_index_hist, center_new, cluster_new, data_index_new):
        detect_drift = Concept_drift()
        r_hist = detect_drift.calcu_r(cluster_hist, center_hist)
        r_new = detect_drift.calcu_r(cluster_new, center_new)
        # dist = detect_drift.calcu_dist(center_hist, center_new)
        dist = detect_drift.calcu_dist_weight(center_hist, data_index_hist, center_new, data_index_new)
        judge_ = detect_drift.judge_drift(dist, r_hist, r_new)

        return judge_


    #semi-supervised learning based on the density cluster algorithm and majority method
    def semi_density_cluster(self, x_train, y_train, data_index):
        semi_y = copy.deepcopy(y_train)
        semi_y_new = semi_y[:, np.newaxis]  # 给每个样本的标签添加一个维度，便于拼接矩阵，形成新的带标记的数据矩阵
        X = copy.deepcopy(x_train)
        with_label_data = np.hstack((X, semi_y_new))  # 在行上进行拼接

        true_label_data = []  # 半监督数据流中带有真实标记的数据集合
        for data in with_label_data:
            if data[-1] != -1:
                true_label_data.append(list(data))

        changed_dataset = []
        last_label = 0

        # 遍历每个簇并记录各个簇中的数据
        for member in data_index.values():  # member为各个簇中所有的样本索引列表
            temp = []                       # 临时保存某个簇中的所有样本 以向量形式
            for index in member:
                temp.append(list(with_label_data[index]))

            lab = [example[-1] for example in temp]  # 一个簇中所有样本的标签集合
            count_label = Counter(lab)  # 给每个簇中的标记计数
            count_label = dict(count_label)  # 转换为字典形式
            count_label = sorted(count_label.items(), key=lambda x: x[1], reverse=True)

            if len(count_label) >= 2:  # 当一个簇内的类别标记不全为-1时
                if count_label[0][0] == -1:
                    last_label = count_label[1][0]  # 除开-1选第二多的标签作为簇中未带标记的样本的标签
                    for data in temp:
                        if data not in true_label_data:  # 真实标记的样本不改动
                            data[-1] = last_label
                    changed_dataset.extend(temp)
                    # print('changed_dataset:', changed_dataset, type(changed_dataset[0]))
                else:
                    last_label = count_label[0][0]  # 当标记的数量多于-1的数量
                    for data in temp:
                        if data not in true_label_data:
                            data[-1] = last_label
                    changed_dataset.extend(temp)
            else:
                for data in temp:  # 当一个簇内的类别标记全为-1时

                    if data not in true_label_data:
                        data[-1] = last_label
                changed_dataset.extend(temp)

        X_list = [example[0:-1] for example in changed_dataset]
        y_list = [example[-1] for example in changed_dataset]

        return X_list, y_list

    #semi-supervised learning based on label propagation
    def semi_label_prop(self,x_train, y_train, data_index):
        #在每个簇内不使用最大类方法，而使用基于图的标签传播算法给未标记样本打上标记

        changed_dataset = []  #for recording the data's new label information

        for member in data_index.values(): #依次处理每个簇
            temp_data = []  #临时记录每个簇中的半监督数据
            temp_label = [] #临时记录每个簇中的半监督标签
            for index in member: #属于同一个簇中的样本索引
                temp_data.append(x_train[index])
                temp_label.append(y_train[index])


            temp_data = np.array(temp_data)
            temp_label = np.array(temp_label)
            # print('temp_data:',temp_data)
            # print('temp_label:',temp_label)
            if len(np.unique(temp_label)) == 1 and temp_label[0] == -1: #当一个簇中全部都是未标记样本
                temp_label[0] = 1
            label_prop_model = LabelPropagation()
            label_prop_model.fit(temp_data, temp_label)
            y_pred = label_prop_model.predict(temp_data)

            X = temp_data
            changed_y = y_pred
            changed_y = changed_y[:, np.newaxis]
            with_label_data = np.hstack((X, changed_y))

            changed_dataset.extend(with_label_data)

        changed_dataset = np.array(changed_dataset)

        X_list = [example[0:-1] for example in changed_dataset]
        y_list = [example[-1] for example in changed_dataset]

        return X_list, y_list

    #semi-supervised learning method in the algorithm
    def semi_method(self, x_train, y_train):
        rho_delta = Rho_multi_Delta(x_train)   # 初始化一个Rho*Delta对象
        rho_multi_delta, sort_rho_m_delta_index, rho, nneigh = rho_delta.cluster()
        change_point = Change_point()          # 初始化一个Change_point对象
        number_cluster = change_point.num_cluster(rho_multi_delta) # 自动确定簇的数量

        #得到聚类信息，聚类中心，每个聚类中的样本索引，都是字典类型
        cluster, cencetr, data_index_in_cluster =\
            rho_delta.cluster_result(number_cluster, sort_rho_m_delta_index, rho, nneigh, x_train)
        # X_list, y_list = self.semi_density_cluster(x_train, y_train, data_index_in_cluster)
        X_list, y_list = self.semi_label_prop(x_train, y_train, data_index_in_cluster)

        return X_list, y_list, cluster, cencetr, data_index_in_cluster

    def detection_cluster_info(self, x_train, y_train):
        rho_delta = Rho_multi_Delta(x_train)  # 初始化一个Rho*Delta对象
        rho_multi_delta, sort_rho_m_delta_index, rho, nneigh = rho_delta.cluster()
        change_point = Change_point()  # 初始化一个Change_point对象
        number_cluster = change_point.num_cluster(rho_multi_delta) # 自动确定簇的数量

        # 得到聚类信息，聚类中心，每个聚类中的样本索引，都是字典类型
        cluster, cencetr, data_index_in_cluster = \
            rho_delta.cluster_result(number_cluster, sort_rho_m_delta_index, rho, nneigh, x_train)

        return cluster, cencetr, data_index_in_cluster



    # detection
    def detect_concept_drift(self, node, brother_node, center, cluster, data_index):
        node.center_new = center
        node.cluster_new = cluster
        node.data_index_new = data_index

        judge_1 = self.judge(node.center_hist, node.cluster_hist, node.data_index_hist,
                             node.center_new, node.cluster_new, node.data_index_new) #判断节点的新旧概念簇集是否产生漂移

        if judge_1 == True and brother_node is not None: #当前节点发生漂移且兄弟节点不为空继续判断它的兄弟节点
            brother_x_train = np.array(brother_node.data_array)  #落入兄弟节点的样本
            brother_y_train = np.array(brother_node.part_label)
            if brother_node.cluster_hist is not None and len(brother_x_train) >= 120:
                bro_cluster, bro_center, bro_data_index \
                    = self.detection_cluster_info(brother_x_train, brother_y_train)  # 给未标记样本打上标记

                brother_node.center_new = bro_center
                brother_node.cluster_new = bro_cluster
                brother_node.data_index_new = bro_data_index

                judge_2 = self.judge(brother_node.center_hist, brother_node.cluster_hist,
                                     brother_node.data_index_hist,brother_node.center_new,
                                     brother_node.cluster_new, brother_node.data_index_new)

                if judge_2 == True:
                    new_leaf = brother_node.parent  # 左右叶子节点都发生漂移定位到对应的父节点，进行剪枝
                    # new_leaf.left_child = None
                    # new_leaf.right_child = None
                    new_leaf = VfdtNode(new_leaf.possible_split_features)

        else:  # 没有检测到漂移，更新
            node.cluster_hist = node.cluster_new
            node.center_hist = node.center_new
            node.data_index_hist = node.data_index_new

    # split node, produce children 分裂节点，生成子树
    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        # k_value = self.k_value
        print('node_split')
        # left = VfdtNode(features, k_value)  #初始化左孩子为一个VfdtNode对象
        # right = VfdtNode(features, k_value) #liu
        left = VfdtNode(features)
        right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)
        #此时就记录了节点的分裂属性和分裂取值，下个样本到来时就调用sort_example()方法就可以划分到孩子节点了

    # predict test example's classification 利用vfdt模型预测样本的类别
    def predict(self, x_test):
        prediction = []  # ndarry
        if isinstance(x_test, np.ndarray) or isinstance(x_test, list):

            for x in x_test:
                leaf = self.root.sort_example(x)
                # print('leaf:',leaf.most_frequent())
                # print(type(leaf))
                prediction.append(leaf.most_frequent())

            return prediction
        else:
            leaf = self.root.sort_example(x_test)
            return leaf.most_frequent()

    def print_tree(self, node):  # 打印树的结构
        if node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)

# 计算评价指标
def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                   labels=np.unique(y_pred)))
    # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无重复元素的元组或者列表
    metrics = pd.DataFrame({'accuracy': accuracy, 'precision': metrics[0], 'recall': metrics[1],
                            'f1': metrics[2]}, index=[row_name])
    # print(row_name)
    # 使用字典创建DataFrame数据
    return metrics



def set_label_ratio(original_data,ratio):
    data_array = copy.deepcopy(original_data)
    total_num = len(data_array)
    label_num = int(total_num * ratio)
    unlabel_num = total_num - label_num
    random_index = random.sample(range(total_num), unlabel_num)
    for j in range(len(random_index)):
        index = random_index[j]
        data_array[index][-1] = -1
    return data_array

def main():
    # dataset_name = ['Sea-abr','Sea-gra','Sine-abr','Sine-gra','HyperPlane-incremental',
    #                 'Random-Tree-abr','Random-Tree-gra']
    dataset_name = ['Electricity.csv']
    # label_ratio = [0.05,0.1,0.2,0.3]
    label_ratio = [0.1]
    # data_number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    # data_number = ['3','4','5','6','7','8','9','10']

    for i_ in range(len(dataset_name)):
        # for n in data_number:
            for j_ in range(len(label_ratio)):
                for k_ in range(1):
                    df = pd.read_csv('Electricity.csv', header=None, sep=',')

                    title = list(df.columns.values)
                    features = title[:-1]  # 获得特征列表
                    rows = df.shape[0]
                    data = df.values
                    # k_value=len(np.unique(data[:,-1]))

                    def split_to_chunk(data, size=1000):  # 将数据流分成大小相同的块，每个块1000个样本
                        s = []
                        for i in range(0, len(data), size):
                            c = data[i:i + size]
                            s.append(c)
                        return s
                    data_block = split_to_chunk(data)  # 得到分块数据
                    tree = Vfdt(features, delta=0.0000001, nmin=200, tau=0.05)  #初始化VFDT
                    i=0
                    acc_list = []
                    for training_set in data_block:
                        i += 1
                        print('Training the {}th data block...'.format(i))
                        if i>1:
                            temp = []
                            x_test = training_set[:, :-1]
                            y_test = list(training_set[:, -1])
                            y_pred = tree.predict(x_test)
                            acc = accuracy_score(y_test, y_pred)
                            temp.append(i)
                            temp.append(acc)
                            acc_list.append(temp)
                            print('ACCURACY: %.4f' % acc)

                        training_data = set_label_ratio(training_set, label_ratio[j_]) #训练数据处理为半监督情况
                        x_train = training_data[:,:-1]
                        y_train = training_data[:,-1]                      #训练数据中不带标记的样本y_train设置为-1
                        for x, y in zip(x_train, y_train):
                            tree.update(x, y)

                    print(acc_list)
                    # acc_list = np.array(acc_list)
                    # acc_list = pd.DataFrame(acc_list)
                    # acc_list.to_csv('D:/FirstWorkResult/Solution_all/'+ dataset_name[i_]+'/'
                    #                 + str(label_ratio[j_]) + '/' + 'Solution_all_' + dataset_name[i_]
                    #                 +'-'+str(k_)+ '_Label_Ratio-'+ str(label_ratio[j_]) + '.csv',  # 结果文件名
                    #                 header=0, index=0)

            # tree.print_tree(tree.root)


if __name__ == "__main__":
    main()

