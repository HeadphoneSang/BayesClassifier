import math
import numpy as np
from scipy.stats import norm


class BayesClassifier:
    def __init__(self,datas):
        self.features = []
        self.priorMap = {}
        self.likeMap = {}
        self.attrs = []
        attrs = datas.columns[1:]
        self.attrs = attrs
        attrsMap = {}
        for i in range(attrs.shape[0]):
            attrsMap[attrs[i]] = i
        self.pre_learn(datas,attrs,attrsMap)

    def pre_learn(self, datas, attrs: list, attrsMap: dict):
        """
        :param attrsMap: 属性索引表 {"属性名":索引}
        :param attrs: 属性列表，[属性索引] = 属性名
        :param datas: pd读取的数据集
        :return: 返回一个np的array和一个类先验概率表还有一个所有特征的所有取值的似然概率表,还有一个每个属性的取值个数表
        """
        features = np.array(datas)
        features = features[:, 1:]
        priorMap = {}
        """
        priorMap = {
            '好': 0.6,
            '坏': 0.4
        }
        """
        likeMap = {}
        """
        likeMap = {
            "色泽":{
                '青涩': [0.1,0.2]
            },
            "根底":{}
        }
        """
        numCntMap = {}  # 记录每个属性的取值
        for i in range(features.shape[1]):
            numCntMap[i] = set()
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                numCntMap[j].add(features[i, j])  # 统计每个属性的取值
        """
        TODO 用set计算每一个属性的每一个取值的似然概率,计算类先验概率
        """
        cCIndex = features.shape[1] - 1  # 类别属性的列下标
        for classValue in numCntMap[cCIndex]:  # 统计类先验概率:P(c)
            priorMap[classValue] = self.get_ClassPrior(attrs[cCIndex], classValue, features, attrsMap,len(numCntMap[cCIndex]))
        classMap = {}  # 统计每个类的取值的个数
        for i in range(features.shape[0]):
            classMap.setdefault(features[i, cCIndex], 0)
            classMap[features[i, cCIndex]] += 1

        for i in range(len(numCntMap) - 1):  # 计算每个属性的每个取值的似然概率
            attr = attrs[i]
            likeMap[attr] = {}
            valueMap = {}
            for j in range(features.shape[0]):  # 统计当前的attr属性的每个取值对应的classValue下的个数
                valueMap.setdefault(features[j, i], {'是': 0, '否': 0})
                valueMap[features[j, i]].setdefault(features[j][cCIndex], 0)
                valueMap[features[j, i]][features[j, cCIndex]] += 1
            for colValue in valueMap.keys():
                likeMap[attr][colValue] = {}
                for classValue in classMap.keys():
                    likeMap[attr][colValue][classValue] = (valueMap[colValue][classValue] + 1) / (
                            classMap[classValue] + len(numCntMap[cCIndex]))
        self.features = features
        self.priorMap = priorMap
        self.likeMap = likeMap
        return features, priorMap, likeMap

    def get_ClassPrior(self, className: str, classValue: str, features: np.array, attrsMap: dict, classValueCnt: int):
        count = 0  # 该classValue样本的数量
        total = features.shape[0]  # 总体样本的数量
        cIndex = attrsMap[className]
        for i in range(features.shape[0]):
            if features[i][cIndex] == classValue:
                count += 1
        return (count + 1) / (total + classValueCnt)

    def get_Post(self, feature: np.array,classValue:str):
        global variance, mean
        ans = self.priorMap[classValue]  # 结果概率
        for i in range(feature.shape[0]):
            if isinstance(feature[i], float):
                mean = np.mean(self.features[:, i])
                variance = np.var(self.features[:, i], ddof=0)
                p = norm.pdf(feature[i], mean, math.sqrt(variance))
                ans *= p
            else:
                attr = self.attrs[i]
                if self.likeMap[attr].get(feature[i]) is None:  # 如果之前没有算过该取值的似然概率
                    self.likeMap[attr].setdefault(feature[i], {'是': 0, '否': 0})
                    state = set()
                    classCnt = 0
                    for j in range(self.features.shape[0]):
                        state.add(self.features[j, i])
                        if self.features[j, self.features.shape[1] - 1] == classValue: classCnt += 1
                    self.likeMap[attr][feature[i]][classValue] = 1 / (classCnt + len(state))
                else:
                    ans *= self.likeMap[attr][feature[i]][classValue]
        return ans
