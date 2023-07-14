import json
import os

import numpy as np
from numpy import nanargmax, sqrt
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import auc, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
import yaml

from preprocess import to_bitmap
from utils import select_stu

model = None
device = None
BITMAP_SIZE = None

sigmoid = torch.nn.Sigmoid()

def threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro",topk=2,threshold=0.5):
    y_score=np.stack(test_df["y_score"])
    y_real=np.stack(test_df["future"])
    best_threshold=0
    if throttle_type=="roc":
        print("throttleing by roc curve")
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        threshold=dict()
        best_threshold_list=[]
        gmeans=dict()
        ix=dict()
        #pdb.set_trace()
        for i in range(BITMAP_SIZE):
            fpr[i], tpr[i], threshold[i] =roc_curve(y_real[:,i],y_score[:,i])
            roc_auc[i] = auc(fpr[i],tpr[i])
            #best:
            gmeans[i] = sqrt(tpr[i]*(1-fpr[i]))
            ix[i]=nanargmax(gmeans[i])
            best_threshold_list.append(threshold[i][ix[i]])
            #print('Dimension: i=%d, Best threshold=%f, G-Mean=%.3f' %(i, threshold[i][ix[i]], gmeans[i][ix[i]]))
        if optimal_type=="indiv":
            best_threshold=best_threshold_list
            y_pred_bin = (y_score-np.array(best_threshold) >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
        elif optimal_type=="micro":
            fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(y_real.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            #best:
            gmeans["micro"] = sqrt(tpr["micro"]*(1-fpr["micro"]))
            ix["micro"]=nanargmax(gmeans["micro"])
            best_threshold=threshold["micro"][ix["micro"]]
            print('Best micro threshold=%f, G-Mean=%.3f' %(best_threshold, gmeans["micro"][ix["micro"]]))
            
            y_pred_bin = (y_score-best_threshold >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
            
    if throttle_type=="f1":
        # print("throttleing by precision-recall curve")
        p = dict()
        r = dict()
        threshold=dict()
        best_threshold_list=[]
        fscore=dict()
        ix=dict()
        
        p["micro"], r["micro"], threshold["micro"]=precision_recall_curve(y_real.ravel(),y_score.ravel())
        fscore["micro"] = (2 * p["micro"] * r["micro"]) / (p["micro"] + r["micro"])
        ix["micro"]=nanargmax(fscore["micro"])
        best_threshold=threshold["micro"][ix["micro"]]
        # print('Best micro threshold=%f, fscore=%.3f' %(best_threshold, fscore["micro"][ix["micro"]]))
        y_pred_bin = (y_score-best_threshold >0)*1
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type=="topk":
        print("throttleing by topk:",topk)
        pred_index = torch.tensor(y_score).topk(topk)[1].cpu().detach().numpy()
        y_pred_bin=[to_bitmap(a,BITMAP_SIZE) for a in pred_index]
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type =="fixed_threshold":
        # print("throttleing by fixed threshold:",threshold)
        best_threshold=threshold
        y_pred_bin = (y_score-np.array(best_threshold) >0)*1
        test_df["predicted"]= list(y_pred_bin)#(all,[length])
    
    return test_df, best_threshold

def model_prediction(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    # print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for data, _ in tqdm(test_loader):
        output = sigmoid(model(data))
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['id', 'cycle', 'addr', 'ip', 'block_address', 'future', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    # print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

def run_val(test_loader, test_df, app_name, model_save_path, option, gpu_id):
    global model
    global device

    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = select_stu(option)

    res = {}

    # print("Validation start")
    test_df = model_prediction(test_loader, test_df, model_save_path)
    df_res, threshold=threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro")
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    res["app"], res["opt_th"], res["p"], res["r"], res["f1"]=[app_name],[threshold],[p],[r],[f1]

    df_res, _ =threshold_throttleing(test_df,throttle_type="fixed_threshold",threshold=0.5)
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    res["p_5"], res["r_5"], res["f1_5"] = [p], [r], [f1]
    
    return res

# def main():
#     global model 
#     global BITMAP_SIZE
    


#     processed_dir = params["system"]["processed"]
#     model_dir = params["system"]["model"]
#     gpu_id = params["system"]["gpu-id"]

#     BITMAP_SIZE = params["hardware"]["bitmap-size"]
#     image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
#     patch_size = (1, image_size[1])
#     num_classes = 2*params["hardware"]["delta-bound"]

#     app_name = params["apps"]["app"]

#     if option == "d":
#         channels = params["model"][f"stu_{option}"]["channels"]
#         model = DenseNetStudent(num_classes, channels)
#     elif option == "r":
#         dim = params["model"][f"stu_{option}"]["dim"]
#         channels = params["model"][f"stu_{option}"]["channels"]
#         model = resnet_tiny(num_classes, channels)
#     elif option == "v":
#         dim = params["model"][f"stu_{option}"]["dim"]
#         depth = params["model"][f"stu_{option}"]["depth"]
#         heads = params["model"][f"stu_{option}"]["heads"]
#         mlp_dim = params["model"][f"stu_{option}"]["mlp-dim"]
#         channels = params["model"][f"stu_{option}"]["channels"]
#         model = TMAP(
#             image_size=image_size,
#             patch_size=patch_size,
#             num_classes=num_classes,
#             dim=dim,
#             depth=depth,
#             heads=heads,
#             mlp_dim=mlp_dim,
#             channels=channels,
#             dim_head=mlp_dim
#         )
#     # elif option == "m":
#     #     model = tch_m  
      

#     test_loader = torch.load(os.path.join(processed_dir, f"test_loader_1"))
#     test_df = torch.load(os.path.join(processed_dir, f"test_df_1"))

#     model_save_path = os.path.join(model_dir, "student.pth")
    
#     res = run_val(test_loader, test_df, app_name, model_save_path)
    
#     file_path = "res/student.json"
#     with open(file_path, "w") as f:
#         json.dump(str(res), f)

#     print(f"Done: results saved at: res/student.json")
    
# if __name__ == "__main__":
