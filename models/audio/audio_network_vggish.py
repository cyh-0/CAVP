import torch
import torch.nn as nn
from models.audio.backbones.vgg import VGG
from models.attn import Mlp
from torchvision.models import resnet18
from models.visual.deeplabv3.encoder_decoder import Backbone
from loguru import logger
from torchvggish import vggish_input, vggish_params
from easydict import EasyDict as edict

cfg = edict()
cfg.BATCH_SIZE = 4 # default 4
cfg.LAMBDA_1 = 0.5 # default: 0.5
cfg.MASK_NUM = 10 # 10 for fully supervised
cfg.NUM_CLASSES = 71 # 70 + 1 background

###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "./torchvggish/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = True #! notice
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "./torchvggish/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../../pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../../pretrained_backbones/pvt_v2_b5.pth"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_AVS_WO_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-111301/checkpoints/checkpoint_29.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-112809/checkpoints/checkpoint_68.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.CROP_IMG_AND_MASK = True
cfg.DATA.CROP_SIZE = 224 # short edge

cfg.DATA.META_CSV_PATH = "../../avsbench_data/metadata.csv" #! notice: you need to change the path
cfg.DATA.LABEL_IDX_PATH = "../../avsbench_data/label2idx.json" #! notice: you need to change the path

cfg.DATA.DIR_BASE = "../../avsbench_data" #! notice: you need to change the path
cfg.DATA.DIR_MASK = "../../avsbench_data/v2_data/gt_masks" #! notice: you need to change the path
cfg.DATA.DIR_COLOR_MASK = "../../avsbench_data/v2_data/gt_color_masks_rgb" #! notice: you need to change the path
cfg.DATA.IMG_SIZE = (224, 224)
###############################
cfg.DATA.RESIZE_PRED_MASK = True
cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (360, 240) # (width, height)

class AudioModel(torch.nn.Module):
    def __init__(self, backbone, pretrain_path, out_plane, num_classes=2):
        super(AudioModel, self).__init__()
        if backbone == "vgg":
            self.backbone = VGGish(cfg, "cuda:0")
            if pretrain_path is not None:
                logger.warning(f'==> Load pretrained VGG parameters from {pretrain_path}')
                # self.backbone.load_state_dict(torch.load(pretrain_path), strict=True)
                self.load_audio_model(path_=pretrain_path)
        else:
            logger.warning(f'==> Load pretrained ResNet18 for Audio')
            self.backbone = resnet18(True)
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.backbone.fc = nn.Linear(512, out_plane)


        self.cls_head = nn.Linear(out_plane, num_classes)

    def forward_cls(self, x):
        return self.cls_head(self.backbone(x))

    def forward(self, x):
        return self.backbone(x)

    def load_audio_model(self, path_):
        # return
        param_dict = torch.load(path_)
        param_ = self.backbone.state_dict()
        out_, in_ = param_["embeddings.4.weight"].shape
        param_dict["embeddings.4.weight"] = torch.nn.init.kaiming_normal_(
            torch.zeros(out_, in_)
        )
        param_dict["embeddings.4.bias"] = torch.zeros(out_)
        self.backbone.load_state_dict(param_dict, strict=True)



class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


# def _spectrogram():
#     config = dict(
#         sr=16000,
#         n_fft=400,
#         n_mels=64,
#         hop_length=160,
#         window="hann",
#         center=False,
#         pad_mode="reflect",
#         htk=True,
#         fmin=125,
#         fmax=7500,
#         output_format='Magnitude',
#         #             device=device,
#     )
#     return Spectrogram.MelSpectrogram(**config)


class VGGish(VGG):
    def __init__(self, cfg, device=None):
        super().__init__(make_layers())
        if cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
            state_dict =  torch.load(cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH)
            super().load_state_dict(state_dict)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device: ", device)
        self.device = device

        self.preprocess = cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL
        self.postprocess = cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA
        if self.postprocess:
            self.pproc = Postprocessor()
            if cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR :
                state_dict = torch.load(cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH)
                # TODO: Convert the state_dict to torch
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )
                self.pproc.load_state_dict(state_dict)
        self.to(self.device)

    def forward(self, x):
        if self.preprocess:
            # print(">>> pre processing...")
            x = self._preprocess(x)
            # print(x.shape)
            x = x.to(self.device)
        x = VGG.forward(self, x)
        if self.postprocess:
            print(">>> post processing...")
            x = self._postprocess(x)
        return x

    def _preprocess(self, x):
        # if isinstance(x, np.ndarray):
        #     x = vggish_input.waveform_to_examples(x, fs)
        batch_num = len(x) # x: audio_path in on batch
        audio_fea_list = []
        for xx in x:
            if isinstance(xx, str):
                xx = vggish_input.wavfile_to_examples(xx) # [5 or 10, 1, 96, 64]
                #! notice:
                if xx.shape[0] != 10:
                    new_xx = torch.zeros(10, 1, 96, 64)
                    new_xx[:xx.shape[0]] = xx
                    audio_fea_list.append(new_xx)
                else:
                    audio_fea_list.append(xx)

        audio_fea = torch.stack(audio_fea_list, dim=0)  #[bs, 10, 1, 96, 64]
        audio_fea = audio_fea.view(batch_num*10, xx.shape[1], xx.shape[2], xx.shape[3]) #[bs*10, 1, 96, 64]
        return audio_fea

    def _postprocess(self, x):
        return self.pproc(x)
