import pathlib
import os
import requests

import torch # tested with PyTorch 1.6
import torchvision as tv # we use torchvision for some pre-trained deep nets. tested with torchvision 0.7
import robustness.datasets
import robustness.model_utils

import wget

class TVPretrainedModel(torch.nn.Module):
  """ a general class for pretrained torchvision models """
  def __init__(self,model_name,input_size):
    """define new model object.
    Args:
      model name (str): the name of the model. Used for retreival and display.
      input_size (int): the image size the model expects to receive as input. A scalar in most cases (e.g., 224)."""
    super().__init__()
    self.model_name=model_name
    self.input_size=input_size # input image sizem

    # image normalization is the same for all torchvision models
    self.normalization_transform=tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]) # https://pytorch.org/docs/stable/torchvision/models.html

    # get imagenet class names
    imagenet_dict_url='https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
    self.idx_to_class_name=eval(requests.get(imagenet_dict_url).text)
    self.class_name_to_idx={v: k for k, v in self.idx_to_class_name.items()} # build inverse dict (class names to indecis)

  def load(self,device=None):
    """download and load torchvision model
    Args:
      device (str/torch.device). Default: cuda:0 if available, otherwise cpu.
    """
    self.add_module("net",
        getattr(tv.models,self.model_name)(pretrained=True))
    if device is None:
      if torch.cuda.is_available():
        self.device=torch.device('cuda',0)
      else:
        self.device=torch.device('cpu')
    else:
      self.device=torch.device(device)
    self.net.to(self.device)
    self.net.eval() # important! move network to inference mode.
    print("loaded {} into {}.".format(self.model_name,str(self.device)))
  def forward(self,im):
    """feed the model with an unnormalized (i.e., intensity in [0,1]), arbitrary sized image.
    Args:
      im (torch.tensor) a 3d (CHW) or 4d (NCHW) image tensor.

    returns class logits and probabilities as tensors."""
    im=im.to(self.device)

    if im.ndim==3: # add a batch dimension, if it's not already there.
      im=im.unsqueeze(0)

    # apply differentiable resizing to bring the input image to the model's expected size
    im=torch.nn.functional.interpolate(im,size=self.input_size,mode='bilinear',align_corners=True) # (interpolate expects NCHW inputs)

    # normalize color channels.
    for i in range(len(im)):
      im[i]=self.normalization_transform(im[i]) # torchvision.transforms.Normalize works only on individual images

    # evaluate model response to the preprocessed image
    logits=self.net(im)
    probabilities=self.probabilities_from_logits(logits)
    return logits, probabilities

  def probabilities_from_logits(self,logits):
    """Transform logits to probabilistic outputs."""
    return torch.nn.Softmax(dim=-1)(logits) # this can be replaced with a smarter readout (e.g. calibrated sigmoids)

# subclasses for particular models
class Resnet50(TVPretrainedModel):
    def __init__(self):
      self=super().__init__('resnet50',224)

class InceptionV3(TVPretrainedModel):
    def __init__(self):
      self=super().__init__('inception_v3',299) # note the different input-image size

# a general class for robustness models
class robustness_pretrained_model(torch.nn.Module):
  def __init__(self,model_name,arch,dataset,model_url,input_size=224):
    super().__init__()
    self.dataset=dataset
    self.model_name=model_name
    self.arch=arch
    self.input_size=input_size
    self.model_url=model_url
    self.model_local_path=os.path.join('robustness_models',self.model_name)+'.pt'

    # get imagenet class names
    imagenet_dict_url='https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
    self.idx_to_class_name=eval(requests.get(imagenet_dict_url).text)
    self.class_name_to_idx={v: k for k, v in self.idx_to_class_name.items()}

  def load(self,device=None):
    ds = getattr(robustness.datasets,self.dataset)(os.path.join('robustness_datasets'+self.dataset))

    # download model
    if not os.path.exists(self.model_local_path):
      pathlib.Path(os.path.dirname(self.model_local_path)).mkdir(parents=True, exist_ok=True)
      wget.download(self.model_url, out=self.model_local_path)
    print(self.arch)
    model, _ = robustness.model_utils.make_and_restore_model(arch=self.arch, dataset=ds,resume_path=self.model_local_path)

    self.add_module("net",model)
    if device is None:
      if torch.cuda.is_available():
        self.device=torch.device('cuda',0)
      else:
        self.device=torch.device('cpu')
    else:
      self.device=torch.device(device)
    self.net.to(self.device)
    self.net.eval() # important! move network to inference mode.
    print("loaded {} into {}.".format(self.model_name,str(self.device)))
  def forward(self,im):
    im=im.to(self.device)

    if im.ndim==3: # add a batch dimension, if it's not already there.
      im=im.unsqueeze(0)

    # apply differentiable resizing to bring the input image to the model's expected size
    im=torch.nn.functional.interpolate(im,size=self.input_size,mode='bilinear',align_corners=True) # (interpolate expects NCHW inputs)

    # ! we do not normalize color channels here because robustness does that within its forward function.

    # evaluate model response to the preprocessed image
    logits=self.net(im,make_adv=False)[0]
    probabilities=self.probabilities_from_logits(logits)
    return logits, probabilities

  def probabilities_from_logits(self,logits):
    return torch.nn.Softmax(dim=-1)(logits)

# subclasses for particular models
# subclasses for particular models
class Resnet_50_l2_eps1(robustness_pretrained_model):
    def __init__(self):
       self=super().__init__(model_name='resnet50_l2_eps1',arch='resnet50',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps1.ckpt',input_size=224)

class Resnet_50_l2_eps3(robustness_pretrained_model):
    def __init__(self):
       self=super().__init__(model_name='resnet50_l2_eps3',arch='resnet50',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt',input_size=224)

class Resnet_50_l2_eps5(robustness_pretrained_model):
    def __init__(self):
       self=super().__init__(model_name='resnet50_l2_eps5',arch='resnet50',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps5.ckpt',input_size=224)

class Wide_Resnet50_2_l2_eps1(robustness_pretrained_model):
    def __init__(self):
      self=super().__init__(model_name='wide_Resnet50_2_l2_eps1',arch='wide_resnet50_2',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_l2_eps1.ckpt',input_size=224)

class Wide_Resnet50_2_l2_eps3(robustness_pretrained_model):
    def __init__(self):
      self=super().__init__(model_name='wide_Resnet50_2_l2_eps3',arch='wide_resnet50_2',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_l2_eps3.ckpt',input_size=224)

class Wide_Resnet50_2_l2_eps5(robustness_pretrained_model):
    def __init__(self):
      self=super().__init__(model_name='wide_Resnet50_2_l2_eps5',arch='wide_resnet50_2',dataset='ImageNet',model_url='https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/wide_resnet50_2_l2_eps5.ckpt',input_size=224)
