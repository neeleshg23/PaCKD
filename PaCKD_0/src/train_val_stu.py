import csv
import os
import sys
import warnings

import yaml
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data_loader import init_dataloader
from utils import select_stu 
from validate_stu import run_val

torch.manual_seed(100)

model = None
optimizer = None
scheduler = None
sigmoid = torch.nn.Sigmoid()

#log = config.Logger()

def train(ep, train_loader, model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        output = sigmoid(model(data))
        loss = F.binary_cross_entropy(output, target, reduction='mean')
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
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, loading, model_save_path, train_loader, test_loader, df_test, app, tsv_path, option, gpu_id):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        print("-------------Model Loaded------------")
        
    best_loss=0
    early_stop = early_stop
    curr_early_stop = early_stop

    metrics_data = []

    for epoch in range(epochs):
        print(f"------- START EPOCH {epoch+1} -------")
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        print((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            print("-------- Save Best Model! --------")
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            print("Early Stop Left: {}".format(curr_early_stop))

        res = run_val(test_loader, df_test, app, model_save_path, option, gpu_id)
        metrics_data.append([epoch+1, train_loss, test_loss, res["opt_th"][0], res["p"][0], res["r"][0], res["f1"][0], res["p_5"][0], res["r_5"][0], res["f1_5"][0]])
        print(f"Epoch {epoch+1:2.0f} Val: opt_threshold={res['opt_th'][0]}, precision={res['p'][0]}, precision_5={res['p_5'][0]}, recall={res['r'][0]}, recall_5={res['r_5'][0]}, accuracy={res['f1'][0]}, accuracy_5={res['f1_5'][0]}")
        
        if curr_early_stop == 0:
            print("-------- Early Stop! --------")
            break

				
    with open(tsv_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Opt_Th', 'P', 'R', 'F1', 'P_5', 'R_5', 'F1_5'])
        writer.writerows(metrics_data)

def main():
    global model
    global optimizer
    global scheduler

    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)

    app = sys.argv[1]
    app_name = app[:-7]
    
    student_model = sys.argv[2]

    gpu_id = sys.argv[3]
    init_dataloader(gpu_id)

    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    results_dir = params["system"]["res"]

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    model = select_stu(student_model)
    print(summary(model))
    model_save_path = os.path.join(model_dir, f"{app_name}.student.{student_model}.pth")
    tsv_path = os.path.join(results_dir, f"{app_name}.student.{student_model}.tsv")

    print("Loading data for student")
    
    df_test = torch.load(os.path.join(processed_dir, f"{app_name}.df.pt"))
            
    train_loader = torch.load(os.path.join(processed_dir, f"{app_name}.train.pt"))
    test_loader = torch.load(os.path.join(processed_dir, f"{app_name}.test.pt"))

    print("Data loaded successfully for stu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loading = False

    run_epoch(epochs, early_stop, loading, model_save_path, train_loader, test_loader, df_test, app, tsv_path, student_model, gpu_id)

if __name__ == "__main__":
    main()
