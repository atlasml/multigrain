dependencies = ['scipy', 'torch', 'torchvision', 'pretrainedmodels']

from multigrain.lib import get_multigrain
import torchvision.transforms as transforms
import torch
import PIL
import urllib.request

def benchmark():
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/multigrain/multigrain_models/joint_3BAA_0.5.pth', 'joint_3BAA_0.5.pth')
    checkpoint = torch.load('joint_3BAA_0.5.pth')
    net = get_multigrain('resnet50')
    net.load_state_dict(checkpoint['model_state'])
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    imagenet.benchmark(
        model=net,
        paper_model_name='MultiGrain + ResNet-50',
        paper_arxiv_id='1902.05509',
        paper_pwc_id='multigrain-a-unified-image-embedding-for',
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1
    )
