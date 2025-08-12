from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
import torch
from torch import nn
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist

from base_model import BaseModel
from models.classification_head import ClassificationHead


class Encoder(Module):
    def __init__(self, **hypparams):
        super().__init__()

        conv_op = convert_dim_to_conv_op(3)  # should return nn.Conv3d

        self.unet = PlainConvUNet(
            input_channels=hypparams["input_channels"],  # e.g. 1
            num_classes=hypparams["num_classes"],  # e.g. 1
            n_stages=5,
            features_per_stage=[32, 64, 128, 256, 512],
            conv_op=conv_op,
            kernel_sizes=[[3, 3, 3]] * 5,  # 3D kernels
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],  # 3D strides
            n_conv_per_stage=[2] * 5,
            n_conv_per_stage_decoder=[2] * 4,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.ReLU,
            nonlin_kwargs={"inplace": True},
            conv_bias=True,
        )
        self.unet.encoder.return_skips = False

        if hypparams["pretrained"]:
            self.unet = load_encoder_from_checkpoint(
                self.unet,
                hypparams,
            )

            if hypparams["finetune_method"] == "linear_probing":
                for _, param in self.unet.named_parameters():
                    param.requires_grad = False

    def forward(self, x):
        # x is (B, C, D, H, W)
        x = self.unet.encoder(x).mean(dim=[2, 3, 4])
        return x


class Encoder_Classifier(BaseModel):
    def __init__(
        self,
        **hypparams,
    ):
        super(Encoder_Classifier, self).__init__(**hypparams)

        self.encoder = Encoder(**hypparams)

        self.cls_head = ClassificationHead(
            512,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls_head(x)

        return x


def load_encoder_from_checkpoint(model, hypparams):
    """
    Loads only the encoder weights from a checkpoint file into an Encoder model.

    Args:
        checkpoint_path (str): Path to the .pth or .pt checkpoint file.
        hypparams (dict): Hyperparameters required for Encoder initialization.
        device (str): Device to map the model and weights ('cpu' or 'cuda').

    Returns:
        Encoder: Encoder model loaded with the checkpoint encoder weights.
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = hypparams["chpt_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get the state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Filter only encoder weights
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if "encoder" in k or "unet.encoder" in k:
            # Remove possible prefixes so keys match
            new_k = k
            if new_k.startswith("model."):
                new_k = new_k[len("model.") :]
            if new_k.startswith("unet."):
                new_k = new_k[len("unet.") :]
            encoder_state_dict[new_k] = v

    # Load weights (non-strict because decoder layers are skipped)
    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model
