from net.backbone import build_backbone
import torch

net = build_backbone('xception')
torch.save(net.state_dict(),'/home/yude/project/pretrained/netmodel.pth')
