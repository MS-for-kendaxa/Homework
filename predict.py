import torch
import torch.nn.functional as F

def predict(input, model_wrapper, output_probs=False):
    model_wrapper.eval()
    outs = model_wrapper(input)
    if output_probs:
        return F.softmax(outs).permute(1,0)
    else:
        return torch.max(outs, dim=-1)[1]