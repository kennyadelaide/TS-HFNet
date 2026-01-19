import os
import yaml
import monai
from monai.networks.nets import (
    UNet, VNet, DynUNet, UNETR, SwinUNETR, SegResNet, AttentionUnet, HighResNet
)
from new_model import UNet3D

class ModelFactory:

    @staticmethod
    def from_config(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return ModelFactory(config)

    def __init__(self, config):
        self.config = config
        self.common_params = config["models"]["common"]

        self.monai_version = tuple(int(x) for x in monai.__version__.split('.')[:2])


    def get_model(self, model_name):
        model_params = self.config["models"].get(model_name, {})

        if model_name == "unet":
            return UNet(
                spatial_dims=3,
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                channels=model_params.get("channels", [16, 32, 64, 128, 256]),
                strides=model_params.get("strides", [2, 2, 2, 2]),
                num_res_units=model_params.get("num_res_units", 2),
                norm=model_params.get("norm", "BATCH"),
            )

        elif model_name == "vnet":
            dropout_prob = model_params.get("dropout_prob", 0.2)
            if self.monai_version >= (1, 2):
                dropout_list = [dropout_prob] * 5

                return VNet(
                    spatial_dims=3,
                    in_channels=self.common_params["in_channels"],
                    out_channels=self.common_params["num_classes"],
                    dropout_prob_up=dropout_list,
                    dropout_prob_down=dropout_list,
                )
            else:

                print(f"VNet使用单个dropout值: {dropout_prob} (MONAI {self.monai_version})")
                return VNet(
                    spatial_dims=3,
                    in_channels=self.common_params["in_channels"],
                    out_channels=self.common_params["num_classes"],
                    dropout_prob_up=dropout_prob,
                    dropout_prob_down=dropout_prob,
                )

        elif model_name == "dynunet":
            return DynUNet(
                spatial_dims=3,
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                kernel_size=model_params["kernel_size"],
                strides=model_params["strides"],
                upsample_kernel_size=model_params["upsample_kernel_size"],
                norm_name=model_params.get("norm_name", "INSTANCE"),
            )

        elif model_name == "unetr":
            return UNETR(
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                img_size=model_params["img_size"],
                feature_size=model_params.get("feature_size", 16),
                hidden_size=model_params.get("hidden_size", 384),
                mlp_dim=model_params.get("mlp_dim", 1536),
                num_heads=model_params.get("num_heads", 8),
                norm_name=model_params.get("norm_name", "instance"),
                res_block=model_params.get("res_block", True),
                dropout_rate=model_params.get("dropout_rate", 0.2),
            )

        elif model_name=="ours":
            return UNet3D(out_channels=4, modal_numbers=4)

        elif model_name == "swin_unetr":


            return SwinUNETR(

                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],


                feature_size=12,
                depths=[2, 2, 2, 2],
                num_heads=[3, 6, 12, 12],
                window_size=[7, 7, 7],


                norm_name="instance",
                drop_rate=0.2,
                attn_drop_rate=0.15,


                use_checkpoint=True,
                spatial_dims=3,

                downsample="merging",
                qkv_bias=True,

            )

        elif model_name == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                init_filters=model_params.get("init_filters", 16),
                blocks_down=model_params.get("blocks_down", [1, 2, 2, 4]),
                blocks_up=model_params.get("blocks_up", [1, 1, 1]),
                dropout_prob=model_params.get("dropout_prob", 0.0),
            )

        elif model_name == "attention_unet":
            return AttentionUnet(
                spatial_dims=3,
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                channels=model_params.get("channels", [16, 32, 64, 128]),
                strides=model_params.get("strides", [2, 2, 2]),
            )

        elif model_name == "highresnet":
            return HighResNet(
                spatial_dims=3,
                in_channels=self.common_params["in_channels"],
                out_channels=self.common_params["num_classes"],
                dropout_prob=model_params.get("dropout_prob", 0.0),
                norm_type="instance",
                bias=True,
            )

        else:
            pass

    def get_all_models(self):
        models = {}
        for model_name in self.config["experiment"]["models_to_run"]:
            try:
                models[model_name] = self.get_model(model_name)

            except Exception as e:
               pass
        return models


if __name__ == "__main__":
    factory = ModelFactory.from_config("config.yaml")
    all_models = factory.get_all_models()
