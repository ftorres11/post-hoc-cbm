import torch
import torch.nn as nn
import numpy as np
import os
osp = os.path
osj = osp.join
from torchvision import transforms
import copy
import pdb


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x

def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if "clip" in backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.out_dir)
        backbone = backbone.eval()
        model = None
    
    elif "resnet18_cream" == backbone_name.lower():
        print('Using ResNet-CREAM')
        from torchvision.models import resnet18
        model = resnet18()
        model.fc = nn.Linear(512,200, bias=True)
        model_sdict = model.state_dict()
        sdict = torch.load(osj('models', 'resnet18_finetuned.pth'))
        for key in sdict.keys():
            model_sdict[key] = sdict[key]
            print('Loaded key: {}'.format(key))
        model.load_state_dict(model_sdict)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cream_mean_pxs = np.array([.485,.456,.406])
        cream_std_pxs = np.array([.229, .224, .225])
        preprocess = transforms.Compose([
                                transforms.CenterCrop(299),
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(mean=cream_mean_pxs,
                                                     std=cream_std_pxs)])

    elif "resnet50" == backbone_name.lower():
        from torchvision.models import resnet50
        model = resnet50(pretrained=True)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        inet_mean_pxs = np.array([.485, .456, .406])
        inet_std_pxs = ([.229, .224, .225])
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=inet_mean_pxs,
                                                              std=inet_std_pxs)])
    elif "resnet18_fmnist" == backbone_name.lower():
        from .resnet_fmnist import resnet18
        model = resnet18()
        sdict = torch.load(osj('models', 'resnet18_fmnist_chkpt_09_14_2025.pth'))
        model.load_state_dict(sdict)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        fmnist_mean = np.array([0.1307])
        fmnist_std = np.array([0.3081])
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(fmnist_mean, fmnist_std)])
    
    elif backbone_name.lower() == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])

    elif backbone_name.lower() == 'inceptionv3_imagenet': 
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3',
                               pretrained=True)
        model_top = copy.deepcopy(backbone)
        model_top.fc = nn.Identity()
        inet_mean_pxs = np.array([.485, .456, .406])
        inet_std_pxs = ([1/.229, 1/.224, 1/.225])
        preprocess = transforms.Compose([transforms.Resize(342),
                                         transforms.CenterCrop(299),
                                         transforms.ToTensor(),
                                         transforms.Normalize(inet_mean_pxs,
                                                              inet_std_pxs)])
    elif backbone_name == 'inceptionv3_concept':
        from .template_model import inception_v3
        backbone = inception_v3(pretrained=False, freeze=False,
                                num_classes=200, aux_logits=True,
                                bottleneck=True, n_attributes=112)
        net_sdict = torch.load(osj('models', 'inceptionv3_concept_weights.pth'))
        backbone.load_state_dict(net_sdict)
        model_top = copy.deepcopy(backbone)
        backbone.all_fc = nn.Identity()
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])


    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


