from sklearn.metrics import classification_report
import json,os
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,f1_score,accuracy_score

class Evaluation(object):
    def __init__(self):
        self.classification_result = []
        self.ranking_result = []
        return 
    
    def classification_eval(self, sim, threshold=0.5, print_result=True):
        label = []
        pred  = []
        pred_prob = []
        for item in sim:
            pred_prob.append(item[2])

            if item[2]>threshold:
                pred.append(1)
            else:
                pred.append(0)


            if item[0]==item[1]:
                label.append(1)
            else:
                label.append(0)

        self.classification_result.append(["Confusion Matrix\n", confusion_matrix(label, pred)])
        self.classification_result.append(["Classification Report\n", classification_report(label, pred, digits=4)])
        self.classification_result.append(["F1", f1_score(label, pred)])
        self.classification_result.append(["Acc", accuracy_score(label, pred)])
        self.classification_result.append(["AUC", roc_auc_score(label, pred_prob)])
        
        if print_result:
            for res in self.classification_result:
                print(res[0], res[1])

        return self.classification_result

    def ranking_eval(self, sim, print_result=True):
        result_dict = {}

        for item in sim:
            if item[0] in result_dict.keys():
                if item[0]==item[1]:
                    result_dict[item[0]].append([1,item[2]])
                else:
                    result_dict[item[0]].append([0,item[2]])
            else:
                if item[0]==item[1]:
                    result_dict[item[0]] = [[1,item[2]]]
                else:
                    result_dict[item[0]] = [[0,item[2]]]
        
        rr = 0
        top_1 = 0
        top_5 = 0
        sample_num = len(result_dict)

        for rlist in result_dict.values():
            rank = sorted(rlist, key= lambda x:x[1], reverse=True)
            rr_count = 0
            for r in rank:
                rr_count += 1
                if r[0]==1:
                    rr += 1/rr_count
                    if rr_count==1:
                        top_1 += 1
                    if rr_count<=5:
                        top_5 += 1
                    break
        
        self.ranking_result.append(["MRR", rr/sample_num])
        self.ranking_result.append(["Pre@1", top_1/sample_num])
        self.ranking_result.append(["Pre@5", top_5/sample_num])

        if print_result:
            for res in self.ranking_result:
                print(res[0], res[1])

        return self.ranking_result