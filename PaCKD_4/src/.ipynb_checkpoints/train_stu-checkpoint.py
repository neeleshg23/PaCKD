import csv
import os
import sys
import warnings
from data_loader import init_dataloader

import yaml

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
from statistics import mean

from validate_stu import run_val
from utils import select_stu, select_tch

torch.manual_seed(100)

model = None
teacher_models = None
optimizer = None
scheduler = None
device = None
Temperature = None

soft_loss = nn.KLDivLoss(reduction="mean", log_target=True)
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, alpha, train_loader, model_save_path, teacher_model):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_preds = model(data)
        
        with torch.no_grad():
            teacher_preds = teacher_model(data)
        
        student_loss = F.binary_cross_entropy(sigmoid(student_preds), target, reduction='mean')

        x_t_sig = sigmoid(teacher_preds / Temperature).reshape(-1)
        x_s_sig = sigmoid(student_preds / Temperature).reshape(-1)

        x_t_p = torch.stack((x_t_sig, 1 - x_t_sig), dim=1)
        x_s_p = torch.stack((x_s_sig, 1 - x_s_sig), dim=1)

        distillation_loss = soft_loss(x_s_p.log(), x_t_p.log())
        loss = alpha * student_loss + (1 - alpha) * distillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = sigmoid(model(data))
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /= len(test_loader)
        return test_loss   

import csv

def run_epoch(epochs, early_stop, alphas, loading, teacher_models_paths, model_save_path, train_loaders, test_loaders, df_test_stu, test_loader_stu, app, tsv_path, option, gpu_id):
    best_loss = 0
    early_stop = early_stop
    curr_early_stop = early_stop
    num_teachers = len(train_loaders)

    if loading:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("-------------Model Loaded------------")

    metrics_data = []

    for epoch in range(epochs):
        print(f"-------------Start Epoch {epoch+1}-------------")
        
        train_losses = []  
        test_losses = []
        
        losses_output = f"Epoch {epoch+1:2.0f}: "
        for i in range(num_teachers):
            tch_model_path = teacher_models_paths[i]
            teacher_model = teacher_models[i].to(device)
            teacher_model.load_state_dict(torch.load(tch_model_path, map_location=device))
            alpha = alphas[i]
            train_loss = train(epoch, alpha, train_loaders[i], model_save_path, teacher_model)
            test_loss = test(test_loaders[i])

            losses_output += f"T{i+1}: TrainL={train_loss:.5f}, TestL={test_loss:.5f}; "
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        # TEST LOSS IS AVERAGE TEST LOSS OVER ALL TEACHERS
        test_loss = mean(test_losses)

        if epoch == 0:
            best_loss = test_loss
        if test_loss <= best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss = test_loss
            losses_output += "Best Model Saved!"
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            losses_output += f"Early Stop Left: {curr_early_stop}!"
        
        res = run_val(test_loader_stu, df_test_stu, app, model_save_path, option, gpu_id)

        metrics_data.append([epoch+1] + train_losses + test_losses + [res["opt_th"][0], res["p"][0], res["r"][0], res["f1"][0], res["p_5"][0], res["r_5"][0], res["f1_5"][0]])
        
        print(f"Epoch {epoch+1:2.0f} Val: opt_th={res['opt_th'][0]}, p={res['p'][0]}, p_5={res['p_5'][0]}, r={res['r'][0]}, r_5={res['r_5'][0]}, f1={res['f1'][0]}, f1_5={res['f1_5'][0]}")
        
        print(losses_output)

        if curr_early_stop == 0:
            losses_output += "Early Stop Triggered!" 
            break
    with open(tsv_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        headers = ['Epoch'] + [f'Train Loss {i+1}' for i in range(num_teachers)] + [f'Test Loss {i+1}' for i in range(num_teachers)] + ['Opt_Th', 'P', 'R', 'F1', 'P_5', 'R_5', 'F1_5']
        writer.writerow(headers)
        writer.writerows(metrics_data)

def main():
    global model
    global teacher_models
    global optimizer
    global scheduler
    global device
    global Temperature

    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)

    app = sys.argv[1]
    app_name = app[:-7]
    
    cluster_option = sys.argv[2]
    alphas = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    
    option = sys.argv[7]
    tch_options = [sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11]]
    
    gpu_id = sys.argv[12]
    init_dataloader(gpu_id)

    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    results_dir = params["system"]["res"]

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]
    Temperature = params["train"]["temperature"]

    model = select_stu(option)
    print(summary(model))

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    df_test_stu = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.df_stu.pt"))
    test_loader_stu = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.test_stu.pt"))

    teacher_models = [select_tch(option) for option in tch_options]
    teacher_models_paths = [os.path.join(model_dir, cluster_option, f"{app_name}.teacher_{i+1}.{option}.pth") for i, option in enumerate(tch_options)]

    train_loaders = []
    test_loaders = []

    for i, _ in enumerate(teacher_models_paths, start=1):
        train_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.train_{i}.pt"))
        test_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.test_{i}.pt"))
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    model_save_path = os.path.join(model_dir, cluster_option, f"{app_name}.student.{option}.{alphas[0]*100}.{alphas[1]*100}.{alphas[2]*100}.pth")
    tsv_path = os.path.join(results_dir, cluster_option, f"{app_name}.student.{option}.{alphas[0]*100}.{alphas[1]*100}.{alphas[2]*100}.tsv")

    loading = False
    
    run_epoch(epochs, early_stop, alphas, loading, teacher_models_paths, model_save_path, train_loaders, test_loaders, df_test_stu, test_loader_stu, app, tsv_path, option, gpu_id)


if __name__ == "__main__":
    main()
