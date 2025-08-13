from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist

from base_model import BaseModel
from models.classification_head import ClassificationHead


class ResEncoder(Module):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder, self).__init__()

        self.res_unet = ResidualEncoderUNet(
            hypparams["input_channels"],
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=hypparams["num_classes"],
        )
        self.res_unet.encoder.return_skips = False

        if hypparams["pretrained"]:
            self.res_unet = load_pretrained_weights(
                self.res_unet,
                hypparams["chpt_path"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.res_unet.named_parameters():
                    param.requires_grad = False

    def forward(self, x):

        x = self.res_unet.encoder(x).mean(dim=[2, 3, 4])

        return x


class ResEncoder_Classifier(BaseModel):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder_Classifier, self).__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        self.cls_head = ClassificationHead(
            320,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls_head(x)

        return x


def load_pretrained_weights(resenc_model, pretrained_weights_file):
    # --- Load checkpoint safely ---
    try:
        if dist.is_initialized():
            saved_model = torch.load(
                pretrained_weights_file,
                map_location=torch.device("cuda", dist.get_rank()),
                weights_only=True,
            )
        else:
            saved_model = torch.load(pretrained_weights_file, weights_only=True)
    except:
        if dist.is_initialized():
            saved_model = torch.load(
                pretrained_weights_file,
                map_location=torch.device("cuda", dist.get_rank()),
                weights_only=False,
            )
        else:
            saved_model = torch.load(pretrained_weights_file, weights_only=False)

    # --- Pick correct dict ---
    if "network_weights" in saved_model:
        pretrained_dict = saved_model["network_weights"]
    elif "state_dict" in saved_model:
        pretrained_dict = saved_model["state_dict"]
    else:
        raise RuntimeError("Checkpoint does not contain network weights or state_dict.")

    # --- Handle DDP / OptimizedModule wrappers ---
    if isinstance(resenc_model, DDP):
        mod = resenc_model.module
    else:
        mod = resenc_model
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # --- Handle input channel mismatches ---
    def adjust_channels(weight_tensor, in_channels_model):
        in_channels_pretrained = weight_tensor.shape[1]
        if in_channels_model == in_channels_pretrained:
            return weight_tensor
        elif in_channels_pretrained < in_channels_model:
            # Repeat and normalize
            return (
                weight_tensor.repeat(1, in_channels_model, 1, 1, 1) / in_channels_model
            )
        else:
            # Slice down to needed channels
            return weight_tensor[:, :in_channels_model, :, :, :]

    # Identify first conv keys we care about
    first_conv_keys = [
        "encoder.stem.convs.0.conv.weight",
        "encoder.stem.convs.0.all_modules.0.weight",
        "decoder.encoder.stem.convs.0.conv.weight",
        "decoder.encoder.stem.convs.0.all_modules.0.weight",
    ]

    # Adjust if needed
    for key in first_conv_keys:
        if key in pretrained_dict and key in model_dict:
            pretrained_dict[key] = adjust_channels(
                pretrained_dict[key], model_dict[key].shape[1]
            )

    # --- Special case for VariableSparkMAETrainer ---
    if "VariableSparkMAETrainer" in pretrained_weights_file:
        pretrained_dict["encoder.stem.convs.0.conv.weight"] = pretrained_dict[
            "encoder.stem.convs.0.all_modules.0.weight"
        ]
        pretrained_dict["decoder.encoder.stem.convs.0.conv.weight"] = pretrained_dict[
            "decoder.encoder.stem.convs.0.all_modules.0.weight"
        ]

    # --- Skip decoder / seg layers ---
    skip_strings_in_pretrained = [
        ".seg_layers.",
        "decoder.stages",
        "decoder.transpconvs",
    ]

    final_pretrained_dict = {}
    for key, v in pretrained_dict.items():
        if key in model_dict and all(s not in key for s in skip_strings_in_pretrained):
            if v.shape != model_dict[key].shape:
                # Handle rare case where only channel count differs
                if (
                    len(v.shape) > 4
                    and v.shape[-3:] == model_dict[key].shape[-3:]
                    and v.shape[0] == model_dict[key].shape[0]
                ):
                    v = v[:, : model_dict[key].shape[1]]
            final_pretrained_dict[key] = v

    # --- Load weights ---
    model_dict.update(final_pretrained_dict)
    mod.load_state_dict(model_dict, strict=False)

    print(
        f"Loaded pretrained weights from {pretrained_weights_file} "
        f"({len(final_pretrained_dict)} parameters matched)"
    )

    return mod
