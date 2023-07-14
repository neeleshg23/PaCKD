import numpy as np
import yaml
from models.l import LSTMModel
from models.m import MLPMixer
from models.r import resnet_tiny, resnet50

def select_stu(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
        
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    
    if option == "r":
        channels = params["model"][f"stu_{option}"]["channels"]
        dim = params["model"][f"stu_{option}"]["dim"]
        return resnet_tiny(num_classes, channels, dim)
    elif option == "m":
        channels = params["model"][f"stu_{option}"]["channels"]
        dim = params["model"][f"stu_{option}"]["dim"]
        depth = params["model"][f"stu_{option}"]["depth"]
        return MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size[1],
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
    elif option == "l":
        input_dim = params["model"][f"stu_{option}"]["input-dim"]
        hidden_dim = params["model"][f"stu_{option}"]["hidden-dim"]
        layer_dim = params["model"][f"stu_{option}"]["layer-dim"]
        output_dim = params["model"][f"stu_{option}"]["output-dim"]
        
        return LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    
def select_tch(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
        
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    
    if option == "r":
        channels = params["model"][f"tch_{option}"]["channels"]
        dim = params["model"][f"tch_{option}"]["dim"]
        return resnet50(num_classes, channels, dim)
    elif option == "m":
        channels = params["model"][f"tch_{option}"]["channels"]
        dim = params["model"][f"tch_{option}"]["dim"]
        depth = params["model"][f"tch_{option}"]["depth"]
        return MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size[1],
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
    elif option == "l":
        input_dim = params["model"][f"tch_{option}"]["input-dim"]
        hidden_dim = params["model"][f"tch_{option}"]["hidden-dim"]
        layer_dim = params["model"][f"tch_{option}"]["layer-dim"]
        output_dim = params["model"][f"tch_{option}"]["output-dim"]
        
        return LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

def select_clu(df_train, df_test, option):
    if option == "a":
        data_train = df_train['past_block_addr'].values
        data_train = np.array(data_train.tolist())

        data_test = df_test['past_block_addr'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test
    
    if option == "d":
        data_train = df_train['past_ip'].values
        data_train = np.array(data_train.tolist())
        data_test = df_test['past_delta'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test
    
    if option == "i":
        data_train = df_train['past_ip'].values
        data_train = np.array(data_train.tolist())

        data_test = df_test['past_ip'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test