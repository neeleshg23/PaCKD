## MacOS Changes To Undo
- `python` -> `python3`
- `Module torchinfo not found` -> Comment out `torchinfo` and `summary()`

## Find OPTIMAL alpha to distill most knowledge

### Binary Alphas {0, 1}
NOKD Stu F1 = 0.9
TchF1s = [0.8, 0.95]
Tch_Alphas = [0, 1], so if TchF1>StuF1 -> Alpha = 1, otherwise, Alpha = 0

### Change Alpha by Student - Teacher Deltas
NOKD Stu F1 = 0.9
TchF1s = [0.8, 0.95]
TchF1_Deltas = [-0.1, +0.05]
Tch_Alphas = [0.4, 0.55]

### Change Alpha by Percent Change in F1 (teacher - student)/student
TchF1_PercentChange = [(0.8-0.9)/0.9, (0.95-0.9)/0.9] = [-11%, +5.56%]
TchF1_Alphas = [0.5(1-0.11), 0.5(1+0.556)] = [.444, .5278]

### Normalize alphas

F1s = [0.8, 0.9, 0.95] on a normal curve where mu=0.9, sigma=Experimental Sigma, then map to
F1s_normal = [-1, 0, 0.5] are the F1 scores on Std. Normal. Curve (mu=0, sigma=1) 
Alpha = [SOME NUMBER 1, 0.5, SOME NUMBER 2], then restandardize at mu=0.5, sigma=SAME SIGMA] <-- potential problem, but F1 range is same as Alpha Range from 0 - 1.

### Check whether F1 normalization exists/Ask Pengmiao 
- Normalization over the range [0, 1], where sigma needs to be calculated ]



