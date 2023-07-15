# PaCKD
PaCKD: Pattern-Clustered Knowledge Distillation for Compressing Memory Access Prediction Models

Cluster options:
- Past Block Address ('a')
- Past Block Address Deltas ('d')
- Past IP ('i')

Models:
- LSTM ('l')
- MLPMixer ('m')
- ResNet ('r')

To run: 
- Import conda env inside of `PaCKD.yaml`
- Change directories inside of `params.yaml`

To preprocess:
- `src/preprocess.py {app} {cluster option} {gpu}`

To train and validate teachers:
- `src/train_tchs.py {app} {cluster option} {model 1} ... {model k} {gpu}`
- `src/validate_tchs.py {app} {cluster option} {model 1} ... {model k} {gpu}`

To train and validate students:
- `src/train_stu.py {app} {cluster option} {alpha tch 1} ... {alpha tch k} {stu model} {tch model 1} ... {tch model k} {gpu}`
