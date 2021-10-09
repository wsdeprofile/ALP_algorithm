import json,os
import joblib
import numpy as np
from geopy.distance import geodesic
import time
from sklearn.svm import SVC


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm==0 or b_norm==0:
        cos = 0
    else:
        cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def get_loc_fea(loc_list_a,loc_list_b):
    count_a = []
    count_b = []

    for item in loc_list_a:
        flag = True
        for c in count_a:
            if geodesic(item, c[0]).m<100:
                c[0][0] = (c[0][0]*c[1]+item[0])/(c[1]+1)
                c[0][1] = (c[0][1]*c[1]+item[1])/(c[1]+1)
                c[1] += 1
                flag = False
                break
        if flag:
            count_a.append([item, 1])

    for item in loc_list_b:
        flag = True
        for c in count_b:
            if geodesic(item, c[0]).m<100:
                c[0][0] = (c[0][0]*c[1]+item[0])/(c[1]+1)
                c[0][1] = (c[0][1]*c[1]+item[1])/(c[1]+1)
                c[1] += 1
                flag = False
                break
        if flag:
            count_b.append([item, 1])


    loc_fea_a = []
    loc_fea_b = []
    shared_a = set()
    shared_b = set()

    for i,loc_a in enumerate(count_a):
        for j,loc_b in enumerate(count_b):
            if geodesic(loc_a[0], loc_b[0]).m<500:
                if i not in shared_a and j not in shared_b:
                    loc_fea_a.append(loc_a[1])
                    loc_fea_b.append(loc_b[1])
                    shared_a.add(i)
                    shared_b.add(j)

    if len(shared_a)==0 or len(shared_b)==0:
        return 0,0

    for i in range(len(count_a)):
        if i not in shared_a:
            loc_fea_a.append(count_a[i][1])
            loc_fea_b.append(0)

    for i in range(len(count_b)):
        if i not in shared_b:
            loc_fea_a.append(0)
            loc_fea_b.append(count_b[i][1])

    loc_fea_a = np.asarray(loc_fea_a)
    loc_fea_b = np.asarray(loc_fea_b)
    f3 = np.sum(np.fmin(loc_fea_a, loc_fea_b))/(1e-5 + np.sum(np.fmax(loc_fea_a, loc_fea_b)))
    f4 = cos_sim(loc_fea_a, loc_fea_b)

    return f3,f4

def get_features(adata, bdata):
    time_fea_a = np.zeros(24+12)
    time_fea_b = np.zeros(24+12)

    loc_list_a = []
    loc_list_b = []

    for record in adata:
        if record[2]==0:
            pass
        else:
            time_data = time.localtime(int(record[2]))
            time_fea_a[time_data.tm_hour] += 1
            time_fea_a[time_data.tm_mon+23] += 1
        
        if record[3]==0:
            pass
        else:
            loc_list_a.append([record[3], record[4]])
        

    for record in bdata:
        if record[1]==0:
            pass
        else:
            time_data = time.localtime(int(record[1]))
            time_fea_b[time_data.tm_hour] += 1
            time_fea_b[time_data.tm_mon+23] += 1

        if record[2]==0:
            pass
        else:
            loc_list_b.append([record[2], record[3]])


    f1 = cos_sim(time_fea_a, time_fea_b)
    f2 = np.sum(np.fmin(time_fea_a, time_fea_b))/(1e-5 + np.sum(np.fmax(time_fea_a, time_fea_b)))
    f3,f4 = get_loc_fea(loc_list_a,loc_list_b)
    

    return [f1,f2,f3,f4]

# 定义算法的类
class MNA(object):
    def __init__(self, args):
        if os.path.exists(args.model_path):
            print("There already exists an MNA model, load it!!!")
            self.model = joblib.load(args.model_path)
        else:
            self.model = SVC(probability=True)
        print("MNA model Initialized!!!")
        return 

    def prepare_data(self,userset,data_a, data_b,args,type):
        x = []
        y = []
        num=0
        for item in userset:
            y.append((item[0], item[1]))
            feature = get_features(data_a[item[0]], data_b[item[1]])
            x.append(feature)
            #num += 1
            #if num%10==0:
                #print(num)

        x,y = np.asarray(x),np.asarray(y)
        if type=='train':
            np.savez('data/train_feature',x=x,y=y)
        elif type=='val':
            np.savez('data/val_feature',x=x,y=y)
        else:
            np.savez('data/test_feature',x=x,y=y)

        return 0

    def preprocessing(self,args):
        print("MNA Load Data!!!")
        dataset = np.load(args.dataset_path, allow_pickle=True).item()
        train = dataset['train']
        val = dataset['val']
        test = dataset['test']

        data_a = {}
        with open(args.location_seq, encoding='utf-8') as f:
            for line in f:
                jdict = json.loads(line.strip())
                uid,ucontent = jdict['id'],jdict['content']
                data_a[uid] = ucontent

        data_b = {}
        with open(args.text_seq, encoding='utf-8') as f:
            for line in f:
                jdict = json.loads(line.strip())
                uid,ucontent = jdict['id'],jdict['content']
                data_b[uid] = ucontent
        
        print("MNA Preprocess Data")
        self.prepare_data(train,data_a, data_b,args,type='train')
        self.prepare_data(val,data_a, data_b,args,type='val')
        self.prepare_data(test,data_a, data_b,args,type='test')

        return 0
    
    def train(self,args):
        print("Train MNA model!!!")
        data = np.load('data/train_feature.npz',allow_pickle=True)
        x,y = data['x'],data['y']
        label = []
        for item in y:
            if item[0]==item[1]:
                label.append(1)
            else:
                label.append(0)
        self.model.fit(x,label)

        if args.save_model:
            print("Save MNA model!!!")
            joblib.dump(self.model, args.model_path)

        return 0

    def test(self,args):
        print("Test MNA model!!!")
        data = np.load('data/test_feature.npz',allow_pickle=True)
        test_x,test_y = data['x'],data['y']
        pred_prob = self.model.predict_proba(test_x)[:,1]    
        #print(pred_prob.shape)

        result = []
        for i in range(len(test_y)):
            result.append([test_y[i][0], test_y[i][1], pred_prob[i]])  
        

        return result


    
