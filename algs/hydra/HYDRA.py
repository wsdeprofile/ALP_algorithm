import joblib
import pandas as pd
import numpy as np
from geopy import distance
import math
import json
from operator import itemgetter
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

stemmer = SnowballStemmer("english")
stop = set(stopwords.words('english'))
word = ['i', 'u', 'you', 'he', 'she', 'his', 'her', 'it', 'its', 'th', '', 'com', 'http', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for i in word:
    stop.add(i)  # 增添停用词


def pretreatment(text):  # 文本预处理
    text = text.lower()
    text = re.sub('[\n]', ' ', text)
    text = re.sub('<.*?>', ' ', text)
    text = re.sub('(@.*?)', ' ', text)
    text = re.sub('[^a-z]', ' ', text)
    text = [stemmer.stem(w) for w in text.split(' ') if w not in stop]  # 去掉停用词并提取词干
    return text


def count_fre(text, k):
    dic = {}
    for word in text:
        dic[word] = dic.get(word, 0) + 1
    words_sorted = sorted(dic.items(), key=itemgetter(1), reverse=True)
    ans = []
    for i in range(k):
        if i < len(words_sorted):
            ans.append(words_sorted[i][0])
        else:
            ans.append("")
    return ans


class HYDRA(object):
    def __init__(self, args):
        self.classifier = svm.SVC(kernel=args.kernel, C=args.C, gamma=args.gamma,
                                  probability=True)
        self.q = args.q
        self.lamda = args.lamda
        self.timeThreshold = args.timeThres
        self.disThreshold = args.distThres
        self.ids = args.ids
        self.text_seq = args.text_seq
        self.location_seq = args.location_seq
        if not args.train:
            # 模型测试过程
            self.model_path = args.model_path  # 只需加载已训练模型

        else:
            # 模型训练过程
            self.save_feature = args.save_feature
            self.feature_path = args.feature_path
            self.model_path = args.model_path

    def dataloder(self):
        """
        加载原始数据
        :return: ids, twitter_ids, flickr_ids, text_seq, location_seq
        """
        ids = pd.read_csv(self.ids, encoding='utf_8_sig')
        with open(self.text_seq) as f:
            text_seq = json.load(f)
        with open(self.location_seq) as f:
            location_seq = json.load(f)
        twitter_ids = ids['twitter_id'].values
        twitter_ids = list(map(lambda x: str(x), twitter_ids))
        # print(twitter_ids)
        text_seq = itemgetter(*twitter_ids)(
            text_seq)  # [[user1[time,lat,lon,text],[time,lat,lon,text]],...[[],[]]] 三维数组，每个用户对应着二维数组（一系列发布行为）

        flickr_ids = ids['flickr_id'].values
        location_seq = itemgetter(*flickr_ids)(
            location_seq)  # [[user1[time,lat,lon,text],[time,lat,lon,text]],...[[],[]]]
        # print(len(location_seq))
        for k in [1, 3, 5]:
            ids['smr_style_' + str(k)] = 0
        for k in [1, 3, 7, 15]:
            ids['smr_time_' + str(k)] = 0
        for k in [1, 3, 7, 15]:
            ids['smr_dis_' + str(k)] = 0
        return ids, twitter_ids, flickr_ids, text_seq, location_seq

    def get_feature(self, ids, twitter_ids, flickr_ids, text_seq, location_seq):
        """

        :param ids: 原始数据dataframe格式 用户id和标签
        :param twitter_ids: twitter用户的id序列
        :param flickr_ids: flickr用户的id序列
        :param text_seq: twitter用户的行为序列
        :param location_seq: flickr用户的行为序列
        :return: ids:得到相似度特征的数据dataframe格式
        """
        print("提取相似度特征")

        def get_user_style(ids):
            """
            计算论文5.2用户文本风格相似度度量
            :param ids：原始用户数据dataframe格式
            :return:ids增加了k=1，3，5时的文本风格相似度，dataframe格式
            """
            print("提取文本风格相似度")
            dict_twitter = {}
            dict_flickr = {}
            for index in range(len(ids)):
                smr_style_list = []
                for k in [1, 3, 5]:
                    text_str = ""
                    if len(text_seq[index]) == 0:
                        dict_twitter[twitter_ids[index]] = ""
                    else:
                        for seq in text_seq[index]:
                            text_str += seq[3]
                        text_str = pretreatment(text_str)
                        dict_twitter[twitter_ids[index]] = count_fre(text_str, k)

                    text_str = ""
                    if len(location_seq[index]) == 0:
                        dict_flickr[flickr_ids[index]] = ""
                    else:
                        for seq in location_seq[index]:
                            text_str += seq[3]
                        text_str = pretreatment(text_str)
                        dict_flickr[flickr_ids[index]] = count_fre(text_str, k)
                        matched_word = 0
                    for i in range(k):
                        if dict_twitter[twitter_ids[index]][i] == dict_flickr[flickr_ids[index]][i]:
                            matched_word += 1
                    smr_style = matched_word / k
                    smr_style_list.append(smr_style)
                ids.loc[index, 'smr_style_1'] = smr_style_list[0]
                ids.loc[index, 'smr_style_3'] = smr_style_list[1]
                ids.loc[index, 'smr_style_5'] = smr_style_list[2]
            # print(ids.head())
            return ids

        def multi_resolution_behavior(ids, q, lamda, timeThreshold, disThreshold):
            """
            计算地点相似性和发文时间相似性
            :param ids:得到用户文本风格相似度后的数据，dataframe格式
            :param q:公式（5）中的q
            :param lamda:sigmoid中的lanmda
            :param timeThreshold:时间阈值
            :param disThreshold:距离阈值
            :return:ids含k=1，3，7，15时的地点相似性和发文时间相似性，dataframe格式
            """
            print("提取地点相似性和发文时间相似性")
            for index in range(len(ids)):
                sim_time_list = []
                sim_dis_list = []
                twitter_id = twitter_ids[index]
                flickr_id = flickr_ids[index]
                twitter_seq = text_seq[index]
                flickr_seq = location_seq[index]
                if len(twitter_seq[0]) == 0 or len(flickr_seq[0]):
                    continue
                time_start = min(twitter_seq[0][0], flickr_seq[0][0])
                time_end = max(twitter_seq[-1][0], twitter_seq[-1][0])

                for k in [1, 3, 7, 15]:
                    smr_time = 0
                    smr_dis = 0
                    twitter_index = 0
                    flickr_index = 0
                    N = (time_end - time_start) / (86400 * k)  # N个时间槽，一天86400秒
                    for time_slot in range(N):
                        # twitter用户的时间与地点在不同时间槽的提取
                        twitter_time_list = []
                        twitter_position_list = []
                        for index in range(twitter_index, len(twitter_seq)):
                            if twitter_seq[index][0] <= time_start + (time_slot + 1) * 86400 * k:
                                twitter_time_list.append(twitter_seq[index][0])
                                twitter_position_list((twitter_seq[index][1], twitter_seq[index][2]))
                            else:
                                twitter_index = index
                                break
                        # flickr用户的时间与地点在不同时间槽的提取
                        flickr_time_list = []
                        flickr_position_list = []
                        for index in range(flickr_index, len(flickr_seq)):
                            if flickr_seq[index][0] <= time_start + (time_slot + 1) * 86400 * k:
                                flickr_time_list.append(flickr_seq[index][0])
                                flickr_position_list.append((flickr_seq[index][1], flickr_seq[index][2]))
                            else:
                                flickr_index = index
                                break
                        time_num = 0
                        dis_num = 0
                        # 时间相似度计算
                        for twitter_time in twitter_time_list:
                            for flickr_time in flickr_time_list:
                                if abs(twitter_time - flickr_time) < timeThreshold:
                                    time_num += 1
                        # 地点相似度计算
                        for twitter_position in twitter_position_list:
                            for flickr_position in flickr_position_list:
                                if distance.distance(twitter_position, flickr_position).kilometers < disThreshold:
                                    dis_num += 1
                        smr_time += pow(time_num, q)
                        smr_dis += pow(dis_num, q)
                    Smr_time = pow(smr_time, 1 / q) / N  # 公式（5）
                    Smr_time_hat = 1 / (1 + math.exp(-lamda * Smr_time))  # sigmoid
                    sim_time_list.append(Smr_time_hat)
                    Smr_dis = pow(smr_time, 1 / q) / N  # 公式（5）
                    Smr_dsi_hat = 1 / (1 + math.exp(-lamda * Smr_dis))  # sigmoid
                    sim_dis_list.append(Smr_dsi_hat)
                ids.loc[index, 'smr_time_1'] = sim_time_list[0]
                ids.loc[index, 'smr_time_3'] = sim_time_list[1]
                ids.loc[index, 'smr_time_7'] = sim_time_list[2]
                ids.loc[index, 'smr_time_15'] = sim_time_list[3]
                ids.loc[index, 'smr_dis_1'] = sim_dis_list[0]
                ids.loc[index, 'smr_dis_3'] = sim_dis_list[1]
                ids.loc[index, 'smr_dis_7'] = sim_dis_list[2]
                ids.loc[index, 'smr_dis_15'] = sim_dis_list[3]
            return ids
        ids = get_user_style(ids)
        ids = multi_resolution_behavior(ids, self.q, self.lamda, self.timeThreshold, self.disThreshold)
        if self.save_feature:
            print("保存特征")
            ids.to_csv(self.feature_path, encoding='utf_8_sig', index=False)
        print(ids.head())
        return ids

    def train(self, train_x, train_y, save):
        # 进行模型的训练
        # 网格搜索
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}
        grid_search = GridSearchCV(self.classifier, param_grid, n_jobs=8, verbose=1)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        self.classifier = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'],
                                  probability=True)
        self.classifier.fit(train_x, train_y)

        if save:
            # 保存模型
            joblib.dump(self.classifier, self.model_path)

    def test(self, test_x, test_y):
        # 加载模型并进行测试
        self.classifier = joblib.load(self.model_path)
        pred_y = self.classifier.predict(test_x)
        print(self.classifier.predict_proba(test_x))
        print(classification_report(test_y, pred_y))
