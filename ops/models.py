from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ops.basic_ops import ConsensusModule, Identity
from ops.transforms import *
import ops.logging as logging
from ops.rsa import RSA

logger = logging.get_logger(__name__)


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, pretrained_parts, modality, dataset,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 transform=None,
                 dropout=0.8,fc_lr5=True,
                 crop_num=1, partial_bn=True, stochastic_depth=0.0, rep_flow=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.dataset = dataset
        self.num_segments = num_segments
        self.pretrained_parts = pretrained_parts
        self.before_softmax = before_softmax
        self.transform = transform
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        self.rep_flow = rep_flow
        self.fc_lr5 = fc_lr5         
        
        # stochastic_depth
        self.stochastic_depth = stochastic_depth
        
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 1
        else:
            self.new_length = new_length

        logger.info("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
    representation flow:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.rep_flow))
        
        if (base_model == 'ResNet'):
            from ops.ResNet import resnet50
            self.base_model = resnet50(
                True,
                num_segments = num_segments,
                kernels=[(1,3,3),(1,3,3),(1,3,3),(1,3,3)],
                conv_modes=['conv','conv','conv','conv'],
                transform=transform,
                groups=[1,1,1,1],
                stochastic_depth = self.stochastic_depth
            )
            self.base_model.last_layer_name = 'fc1'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]            
            feature_dim = self._prepare_tsn(num_class)   
        else:    
            raise ValueError('Unknown base model: {}'.format(args.arch))

        

        if self.modality == 'Flow':
            logger.info("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            logger.info("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            logger.info("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            logger.info("Done. RGBDiff model ready.")
        elif self.modality == 'RGB' and self.rep_flow:
            logger.info("Converting the ImageNet model to a rep. flow init model")
#             self.base_model = self._construct_repflow_model(self.base_model)
            logger.info("Done. RepFlow model ready...")
            
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        
        
    def _prepare_tsn(self, num_class):    
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_channels
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0,bias=True))  
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0,bias=True)

        std = 0.001
        
        if self.new_fc is None:
            xavier_uniform_(getattr(self.base_model, self.base_model.last_layer_name).weight)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)   
            
        return feature_dim

            
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            logger.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():                
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            logger.info("No BN layer Freezing.")
    
    
    def partialBN(self, enable):
        self._enable_pbn = enable
    
    
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, RSA):
                for name, param in m.named_parameters():
                    if name in ['P1', 'I']:
                        normal_weight.append(param)
            if isinstance(m, (torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.ConvTranspose2d,torch.nn.ConvTranspose3d,torch.nn.Linear)):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        
        print(normal_bias)
        
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]                

    

    def forward(self, input):
             
        #############################################################
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length 
        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # input.size(): [32, 9, 224, 224]
        # after view() func: [96, 3, 224, 224]
        if (self.base_model_name in ['ResNet']):    # [B, C, T, W, H]
            before_permute = input.view((-1, self.num_segments, sample_len) + input.size()[-2:]).contiguous()
            input_var = before_permute.permute(0,2,1,3,4).contiguous()
        else:       
            input_var = input.view((-1, sample_len) + input.size()[-2:])


        base_out = self.base_model(input_var)
        
        # zc comments
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
            
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        
        # zc comments end
        base_out = base_out.view((-1, (self.num_segments)) + base_out.size()[1:])
        
        output = base_out.mean(dim=1, keepdim=True)            

        # output after squeeze(1): [32, 101], forward() returns size: [batch_size, num_class]
        return output.squeeze(3).squeeze(1)  #, recon_loss, smooth_loss#, repflow
    
    
    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_repflow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        
        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernels = params[0].data.repeat(1,2,1,1)
        new_conv = nn.Conv2d(3, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name
        
        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model
    
    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip2(dataset=self.dataset,is_flow=False)])
#                                                    GroupRandomHorizontalFlip(is_flow=False)])        
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
