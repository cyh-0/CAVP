import torch
import torch.nn as nn
import torchvision.models as models
from models.pvt import pvt_v2_b5
from models.TPAVI import TPAVIModule
import pdb


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.vis_dim = vis_dim

        self.encoder_backbone = pvt_v2_b5()
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        if self.training:
            self.initialize_pvt_weights()
        
        self.num_frame = 1


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, self.num_frame, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x

    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, self.num_frame, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, audio_feature=None):
        x1, x2, x3, x4 = self.encoder_backbone(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # shape for pvt-v2-b5
        # BF x  64 x 56 x 56
        # BF x 128 x 28 x 28
        # BF x 320 x 14 x 14
        # BF x 512 x  7 x  7

        conv1_feat = self.conv1(x1)    # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)    # BF x 256 x 28 x 28
        conv3_feat = self.conv3(x3)    # BF x 256 x 14 x 14
        conv4_feat = self.conv4(x4)    # BF x 256 x  7 x  7
        # print(conv1_feat.shape, conv2_feat.shape, conv3_feat.shape, conv4_feat.shape)

        feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
        a_fea_list = [None] * 4

        if len(self.tpavi_stages) > 0:
            if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
                raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
                    tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
            for i in self.tpavi_stages:
                tpavi_count = 0
                conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
                if self.tpavi_vv_flag:
                    conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
                    conv_feat += conv_feat_vv
                    tpavi_count += 1
                if self.tpavi_va_flag:
                    conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
                    conv_feat += conv_feat_va
                    tpavi_count += 1
                    a_fea_list[i] = a_fea
                conv_feat /= tpavi_count
                feature_map_list[i] = conv_feat # update features of stage-i which conduct non-local

        conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
        # print(pred.shape)

        return pred, feature_map_list, a_fea_list


    def initialize_pvt_weights(self,):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        # pdb.set_trace()


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    # model = Pred_endecoder(channel=256)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
    # output = model(imgs)
    output = model(imgs, audio)
    pdb.set_trace()