import torch
def get_repartition_gpu():
        x = torch.cuda.device_count()
        print("Number of gpus:",x)
        if x==2: return [0,1],[0,1]
        elif x==3: return [0,1],[1,2]
        elif x>3: return [0,1,2,3],[0,2,1,3] 