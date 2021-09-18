from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import json
import joblib
from geopy import distance
from collections import defaultdict


# 定义算法的类
class STUL(object):
    def __init__(self, args):
        # 进行类的初始化
        self.classifier = svm.SVC(kernel=args.kernel, C=args.C, gamma=args.gamma,
                                  probability=True)
        if not args.train:
            # 模型测试过程
            self.model_path = args.model_path  # 只需加载已训练模型

        else:
            # 模型训练过程
            self.save_feature = args.save_feature
            self.feature_path = args.feature_path
            self.model_path = args.model_path

    def get_features(self, ids, twitter_seq, flickr_seq, distThres, timeThres, dc, k):
        """
        :param ids: 没有similarity的数据 dataframe形式：header:twitter_id, flickr_id, label
        :param twitter_seq: twitter的用户形式信息 字典形式：id:[[time, lat, lon, post],[],...,]
        :param flickr_seq: flickr 字典形式：id:[[time, lat, lon, post],[],...,]
        :param distThres: 提取stay_points的距离阈值
        :param timeThres: 提取stay_points的时间阈值
        :param dc: 论文4.1求p值的dc
        :param k: 论文4.1 top-k个聚类中心
        :return: ids 含有similarity特征的数据 dataframe形式：header:twitter_id, flickr_id, similarity, label
        """
        # 提高代码的复用性，可以自己定义一些方法，比如特征计算的方法
        def getTimeIntervalOfPoints(pi, pj):
            return pj - pi

        def stayPointExtraction(points, distThres=distThres, timeThres=timeThres):
            stayPointList = []
            pointNum = len(points)
            i = 0
            while i < pointNum:
                j = i + 1
                while j < pointNum:

                    if distance.distance((points[i][1], points[i][2]),
                                         (points[j][1], points[j][2])).kilometers > distThres:
                        # points[j] has gone out of bound thus it should not be counted in the stay points.
                        if getTimeIntervalOfPoints(points[i][0], points[j - 1][0]) > timeThres:
                            stayPointList.extend(points[i:j])
                        break
                    j += 1
                i = j
            return stayPointList

        # 加载用户id和标签（用户id， label）
        ids = pd.read_csv(ids, encoding='utf_8_sig')
        print(ids.head())

        # 加载用户行为信息
        print("加载用户行为信息================>")
        with open(twitter_seq, 'r+', encoding='utf_8_sig') as f1:
            twitter_seq = json.load(f1)  # key 为id，值为行为列表[[time,lat,lon,post],[]...[]]
        with open(flickr_seq, 'r+', encoding='utf_8_sig') as f2:
            flickr_seq = json.load(f2)
        print("加载完毕================>")

        def get_stay_pionts():
            """
            从原始points中得到stay_points
            :return: dict_twitter_staypoints, dict_flickr_staypoints 字典形式：id:[point1,point2....]
            """
            print("获取stay_points=============>")
            dict_twitter_staypoints = defaultdict(list)
            dict_flickr_staypoints = defaultdict(list)

            visit_twitter = {}  # 防止重复添加
            visit_flickr = {}

            for index, item in ids.iterrows():
                twitter_id = str(item['twitter_id'])
                flickr_id = item['flickr_id']

                twitter_points = []
                flickr_points = []

                if not visit_twitter.get(twitter_id, False):  # 如果该用户没添加过
                    visit_twitter[twitter_id] = True
                    # 获取gps信息，去掉post信息，XXX_gps[0:3]为[time,lat,lon]，XXX_points为[[time,lat,lon],[]....[]]
                    for twitter_gps in twitter_seq[twitter_id]:
                        twitter_points.append(twitter_gps[0:3])
                    # 获取stay_points，dict['user_id'] = [[t,lat,lon],[],...[]]
                    dict_twitter_staypoints[twitter_id] = stayPointExtraction(twitter_points)
                    if len(dict_twitter_staypoints[twitter_id]) == 0:
                        dict_twitter_staypoints[twitter_id].append([])

                if not visit_flickr.get(flickr_id, False):
                    visit_flickr[flickr_id] = True
                    for flickr_gps in flickr_seq[flickr_id]:
                        flickr_points.append(flickr_gps[0:3])
                    dict_flickr_staypoints[flickr_id] = stayPointExtraction(flickr_points)
                    if len(dict_flickr_staypoints[flickr_id]) == 0:
                        dict_flickr_staypoints[flickr_id].append([])
            # with open(r'data\twitter_stay_points.json', 'w') as f1:
            #     json.dump(dict_twitter_staypoints, f1)
            # with open(r'data\flickr_stay_points.json', 'w') as f2:
            #     json.dump(dict_flickr_staypoints, f2)
            return dict_twitter_staypoints, dict_flickr_staypoints

            print("获取完毕=============>")

        def func(i):
            return i[1]

        def get_clusters():
            """
            得到聚类
            :return:twitter_center_and_nocenter, flickr_center_and_nocente 字典形式: id:[[point1, point2...], cluster,....]
            """
            # print("计算论文呢里的p=================>")

            # with open(r'\data\twitter_stay_candidate_points.json', 'r+', encoding='utf_8_sig') as f1:
            #     dict_twitter_staypoints = json.load(f1)  # key 为id，值为行为列表[[time,lat,lon,post],[]...[]]
            # with open(r'\data\flickr_stay_candidate_points.json', 'r+', encoding='utf_8_sig') as f2:
            #     dict_flickr_staypoints = json.load(f2)
            dict_twitter_staypoints, dict_flickr_staypoints = get_stay_pionts()
            dict_flickr_p = defaultdict(list)  # 存储论文里的p
            dict_twitter_p = defaultdict(list)
            visit_twitter = {}  # 防止重复添加
            visit_flickr = {}
            for index, id in ids.iterrows():
                print(index)
                twitter_id = str(id['twitter_id'])
                filckr_id = id['flickr_id']

                if visit_twitter.get(twitter_id, False) == False:  # 如果该用户没添加过
                    visit_twitter[twitter_id] = True
                    len_twitter = len(dict_twitter_staypoints[twitter_id])
                    print("len_twitter:", len_twitter)
                    for i in range(len_twitter):
                        num = 0
                        pi = dict_twitter_staypoints[twitter_id][i]  # 第i个Point
                        for j in range(len_twitter):
                            if i != j:
                                if distance.distance((pi[1], pi[2]), (dict_twitter_staypoints[twitter_id][j][1],
                                                                      dict_twitter_staypoints[twitter_id][j][
                                                                          2])).kilometers < dc:
                                    num += 1
                        dict_twitter_p[twitter_id].append(num)

                if visit_flickr.get(filckr_id, False) == False:  # 如果该用户没添加过
                    visit_flickr[filckr_id] = True
                    len_flickr = len(dict_flickr_staypoints[filckr_id])
                    print("len_flickr:", filckr_id, len_flickr)
                    for i in range(len_flickr):
                        num = 0
                        pi = dict_flickr_staypoints[filckr_id][i]  # 第i个Point
                        for j in range(len_flickr):
                            if i != j:
                                if distance.distance((pi[1], pi[2]), (
                                        dict_flickr_staypoints[filckr_id][j][1],
                                        dict_flickr_staypoints[filckr_id][j][2])) < dc:
                                    num += 1
                        dict_flickr_p[filckr_id].append(num)
            # with open(r'data\twitter_p.json', 'w') as f1:
            #     json.dump(dict_twitter_p, f1)
            # with open(r'data\flickr_p.json', 'w') as f2:
            #     json.dump(dict_flickr_p, f2)
            print("计算完毕=====================>")

            print("计算论文里的δ==================>")
            # with open(r'\data\twitter_p.json', 'r+', encoding='utf_8_sig') as f1:
            #     dict_twitter_p = json.load(f1)  # key 为id，值为行为列表[[time,lat,lon,post],[]...[]]
            # with open(r'\data\flickr_p.json', 'r+', encoding='utf_8_sig') as f2:
            #     dict_flickr_p = json.load(f2)

            visit_twitter = {}
            visit_flickr = {}
            dict_twitter_ysm = defaultdict(list)
            dict_flickr_ysm = defaultdict(list)
            for index, id in ids.iterrows():
                print(index)
                twitter_id = str(id['twitter_id'])
                filckr_id = id['flickr_id']
                if not visit_twitter.get(twitter_id, False):
                    visit_twitter[twitter_id] = True
                    for i in range(len(dict_twitter_p[twitter_id])):
                        pi = dict_twitter_p[twitter_id][i]
                        di = dict_twitter_staypoints[twitter_id][i]
                        minysm = 1e10
                        maxysm = 0
                        flag = False
                        for j in range(i):
                            if j != i:
                                if dict_twitter_p[twitter_id][j] > pi:
                                    flag = True
                                dis = distance.distance((di[1], di[2]), (dict_twitter_staypoints[twitter_id][j][1],
                                                                         dict_twitter_staypoints[twitter_id][j][
                                                                             2])).kilometers
                                minysm = min(minysm, dis)
                                maxysm = max(maxysm, dis)
                        if flag:
                            dict_twitter_ysm[twitter_id].append(minysm)
                        else:
                            dict_twitter_ysm[twitter_id].append(maxysm)

                if not visit_flickr.get(filckr_id, False):
                    visit_flickr[filckr_id] = True

                    for i in range(len(dict_flickr_p[filckr_id])):
                        pi = dict_flickr_p[filckr_id][i]
                        di = dict_flickr_staypoints[filckr_id][i]
                        minysm = 1e10
                        maxysm = 0
                        flag = False
                        for j in range(i):
                            if j != i:
                                if dict_flickr_p[filckr_id][j] > pi:
                                    flag = True
                                dis = distance.distance((di[1], di[2]), (dict_flickr_staypoints[filckr_id][j][1],
                                                                         dict_flickr_staypoints[filckr_id][j][
                                                                             2])).kilometers
                                minysm = min(minysm, dis)
                                maxysm = max(maxysm, dis)
                        if flag:
                            dict_flickr_ysm[filckr_id].append(minysm)
                        else:
                            dict_flickr_ysm[filckr_id].append(maxysm)
            # with open(r'data\twitter_ysm.json', 'w') as f1:
            #     json.dump(dict_twitter_ysm, f1)
            # with open(r'data\flickr_ysm.json', 'w') as f2:
            #     json.dump(dict_flickr_ysm, f2)
            print("计算完毕==================>")

            print("计算聚类==================>")
            # with open(r'\data\twitter_ysm.json', 'r+', encoding='utf_8_sig') as f1:
            #     dict_twitter_ysm = json.load(f1)  # key 为id，值为行为列表[[time,lat,lon,post],[]...[]]
            # with open(r'\data\flickr_ysm.json', 'r+', encoding='utf_8_sig') as f2:
            #     dict_flickr_ysm = json.load(f2)

            twitter_clusters = defaultdict(list)
            twitter_center_and_nocenter = defaultdict(list)

            for twitter_id in dict_twitter_p.keys():
                for i in range(len(dict_twitter_p[twitter_id])):
                    twitter_clusters[twitter_id].append(
                        (i, dict_twitter_p[twitter_id][i] * dict_twitter_ysm[twitter_id][i]))
                twitter_clusters[twitter_id].sort(key=func, reverse=True)
                list_centers = [[] for _ in range(k)]
                for item in twitter_clusters[twitter_id]:
                    min_dis = 1e10
                    for center in range(k):
                        center_index = twitter_clusters[twitter_id][center][0]
                        dis = distance.distance((dict_twitter_staypoints[twitter_id][item[0]][1],
                                                 dict_twitter_staypoints[twitter_id][item[0]][2]), (
                                                    dict_twitter_staypoints[twitter_id][center_index][1],
                                                    dict_twitter_staypoints[twitter_id][center_index][2])).kilometers
                        if min_dis > dis:
                            min_dis = dis
                            center_index_follow = center
                    list_centers[center_index_follow].append(dict_twitter_staypoints[twitter_id][item[0]])
                for center in range(k):
                    center_index = twitter_clusters[twitter_id][center][0]
                    list_centers[center].append(dict_twitter_staypoints[twitter_id][center_index])
                twitter_center_and_nocenter[twitter_id] = list_centers

            flickr_clusters = defaultdict(list)
            flickr_center_and_nocenter = defaultdict(list)
            for flickr_id in dict_flickr_p.keys():
                for i in range(len(dict_flickr_p[flickr_id])):
                    flickr_clusters[flickr_id].append((i, dict_flickr_p[flickr_id][i] * dict_flickr_ysm[flickr_id][i]))
                flickr_clusters[flickr_id].sort(key=func, reverse=True)
                list_centers = [[] for _ in range(k)]
                for item in flickr_clusters[flickr_id]:
                    min_dis = 1e10
                    for center in range(k):
                        center_index = flickr_clusters[flickr_id][center][0]
                        dis = distance.distance((dict_flickr_staypoints[flickr_id][item[0]][1],
                                                 dict_flickr_staypoints[flickr_id][item[0]][2]), (
                                                    dict_flickr_staypoints[flickr_id][center_index][1],
                                                    dict_flickr_staypoints[flickr_id][center_index][2])).kilometers
                        if min_dis > dis:
                            min_dis = dis
                            center_index_follow = center
                    list_centers[center_index_follow].append(dict_flickr_staypoints[flickr_id][item[0]])
                for center in range(k):
                    center_index = flickr_clusters[flickr_id][center][0]
                    list_centers[center].append(dict_flickr_staypoints[flickr_id][center_index])
                flickr_center_and_nocenter[flickr_id] = list_centers
            print("计算完毕==================>")
            return twitter_center_and_nocenter, flickr_center_and_nocenter

        def get_similarity():
            """
            得到用户间的相似度
            :return: ids dataframe形式： header:twitter_id, flickr_id,similarity, label
            """
            # 若生成中间文件，可直接加载中间文件
            # with open(r'\data\twitter_clusters.json', 'r+', encoding='utf_8_sig') as f1:
            #     dict_twitter_clusters = json.load(f1)  # key 为id，值为行为列表[[time,lat,lon,post],[]...[]]
            # with open(r'\data\flickr_clusters.json', 'r+', encoding='utf_8_sig') as f2:
            #     dict_flickr_clusters = json.load(f2)

            dict_twitter_clusters, dict_flickr_clusters = get_clusters()
            dict_twitter_cluster_represent = defaultdict(list)
            dict_flickr_cluster_represent = defaultdict(list)

            print("得到点集代表...........")
            # twitter_clusters点集代表,用平均值表示
            for key in dict_twitter_clusters.keys():
                # print(len(dict_twitter_clusters[key]))
                for i in range(len(dict_twitter_clusters[key])):
                    col_totals = [sum(x) for x in zip(*dict_twitter_clusters[key][i])]
                    num_points = len(dict_twitter_clusters[key][i])
                    dict_twitter_cluster_represent[key].append((col_totals[0] / num_points, col_totals[1] / num_points))
                # print(len(dict_twitter_cluster_represent[key]))

            # flickr_clusters点集代表，用平均值表示
            for key in dict_flickr_clusters.keys():
                for i in range(len(dict_flickr_clusters[key])):
                    col_totals = [sum(x) for x in zip(*dict_flickr_clusters[key][i])]
                    num_points = len(dict_flickr_clusters[key][i])
                    dict_flickr_cluster_represent[key].append((col_totals[0] / num_points, col_totals[1] / num_points))
                # print(len(dict_flickr_cluster_represent[key]))
            print("完成===================>")
            # 得到点集相似度
            print("计算点集相似度...............")
            dict_twitter_sim = defaultdict(list)
            for key in dict_twitter_cluster_represent.keys():
                for cluster_represnt in dict_twitter_cluster_represent[key]:
                    sim = 0
                    for key_other in dict_twitter_cluster_represent.keys():
                        if key_other == key:
                            continue
                        else:
                            for cluster_represnt_other in dict_twitter_cluster_represent[key_other]:
                                dis = distance.distance((cluster_represnt[0], cluster_represnt[1]),
                                                        (cluster_represnt_other[0],
                                                         cluster_represnt_other[1])).kilometers
                                sim += 1 / dis if dis > 0 else 1  # 点集距离倒数表示相似度
                    dict_twitter_sim[key].append(sim)

            dict_flickr_sim = defaultdict(list)
            for key in dict_flickr_cluster_represent.keys():
                for cluster_represnt in dict_flickr_cluster_represent[key]:
                    sim = 0
                    for key_other in dict_flickr_cluster_represent.keys():
                        if key_other == key:
                            continue
                        else:
                            for cluster_represnt_other in dict_flickr_cluster_represent[key_other]:
                                dis = distance.distance((cluster_represnt[0], cluster_represnt[1]),
                                                        (cluster_represnt_other[0],
                                                         cluster_represnt_other[1])).kilometers
                                sim += 1 / dis if dis > 0 else 1  # 点集距离倒数表示相似度
                    dict_flickr_sim[key].append(sim)
            print("计算完成=====================>")
            # 得到权重
            print("计算点集权重..............")
            dict_twitter_weight = defaultdict(list)
            for key in dict_twitter_sim.keys():
                # print(len(dict_twitter_sim[key]))
                N = max(dict_twitter_sim[key])
                # print(N)

                f_list = list(map(lambda x: N / (1 + x), dict_twitter_sim[key]))
                # print(f_list)
                f_sum = sum(f_list)
                weight_list = list(map(lambda x: x / f_sum, f_list))
                # print(weight_list)
                dict_twitter_weight[key] = weight_list
                # print(dict_twitter_weight[key])

            dict_flickr_weight = defaultdict(list)
            for key in dict_flickr_sim.keys():
                # print(dict_twiiter_sim[key])
                N = 1 if len(dict_twitter_sim[key]) == 0 else max(dict_twitter_sim[key])
                f_list = list(map(lambda x: N / (1 + x), dict_flickr_sim[key]))
                # print(f_list)
                f_sum = sum(f_list) if sum(f_list) != 0 else 1
                # print(f_sum)
                weight_list = list(map(lambda x: x / f_sum, f_list))
                # print(weight_list)
                dict_flickr_weight[key] = weight_list
                # print(dict_flickr_weight[key])
            print("计算完成===================》")
            # 得到用户的相似度
            print("计算用户间的相似度...............")
            ids['simlarity'] = 0
            for index, item in ids.iterrows():
                twitter_id = str(item['twitter_id'])
                flickr_id = item['flickr_id']
                sim_sum = 0
                for i in range(len(dict_twitter_sim[twitter_id])):
                    for j in range(len(dict_flickr_sim[flickr_id])):
                        dis = distance.distance(dict_twitter_cluster_represent[twitter_id][i],
                                                dict_flickr_cluster_represent[flickr_id][j]).kilometers
                        dis_inverse = 1 / dis if dis > 0 else 1
                        sim_sum += dis_inverse * dict_twitter_weight[twitter_id][i] * dict_flickr_weight[flickr_id][j]
                ids.iloc[index, 3] = sim_sum
            print("计算完成==============>")
            if self.save_feature:
                print("保存")
                ids.to_csv(self.feature_path, encoding='utf_8_sig')
            return ids

        return get_similarity()

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
