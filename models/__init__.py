from .mednext import MedNeXt, MedNeXt_small
from .unet3d import (
    UNet,
    UNet_small_4f, UNet_small_3f, UNet_small_2f, UNet_small_1f,
    LUNet_small_4f, LUNet_small_3f, LUNet_small_2f, LUNet_small_1f,
    UNet_large_24f, UNet_large_12f, UNet_large_6f,
    LUNet_large_24f, LUNet_large_12f, LUNet_large_6f,
)
from .unetplusplus import (
    UNetPlusPlus,
    UNetPP_small_4f, UNetPP_small_3f, UNetPP_small_2f, UNetPP_small_1f,
    LUNetPP_small_4f, LUNetPP_small_3f, LUNetPP_small_2f, LUNetPP_small_1f,
    UNetPP_large_24f, UNetPP_large_12f, UNetPP_large_6f,
    LUNetPP_large_24f, LUNetPP_large_12f, LUNetPP_large_6f,
)

_REGISTRY = {
    # --- UNet small ---
    "unet_small_4f":  UNet_small_4f,
    "unet_small_3f":  UNet_small_3f,
    "unet_small_2f":  UNet_small_2f,
    "unet_small_1f":  UNet_small_1f,
    # --- LUNet small ---
    "lunet_small_4f": LUNet_small_4f,
    "lunet_small_3f": LUNet_small_3f,
    "lunet_small_2f": LUNet_small_2f,
    "lunet_small_1f": LUNet_small_1f,
    # --- UNet large ---
    "unet_large_24f": UNet_large_24f,
    "unet_large_12f": UNet_large_12f,
    "unet_large_6f":  UNet_large_6f,
    # --- LUNet large ---
    "lunet_large_24f": LUNet_large_24f,
    "lunet_large_12f": LUNet_large_12f,
    "lunet_large_6f":  LUNet_large_6f,
    # --- MedNeXt ---
    "mednext_small": MedNeXt_small,
    # --- UNet++ small ---
    "unetpp_small_4f":  UNetPP_small_4f,
    "unetpp_small_3f":  UNetPP_small_3f,
    "unetpp_small_2f":  UNetPP_small_2f,
    "unetpp_small_1f":  UNetPP_small_1f,
    # --- LUNet++ small ---
    "lunetpp_small_4f": LUNetPP_small_4f,
    "lunetpp_small_3f": LUNetPP_small_3f,
    "lunetpp_small_2f": LUNetPP_small_2f,
    "lunetpp_small_1f": LUNetPP_small_1f,
    # --- UNet++ large ---
    "unetpp_large_24f": UNetPP_large_24f,
    "unetpp_large_12f": UNetPP_large_12f,
    "unetpp_large_6f":  UNetPP_large_6f,
    # --- LUNet++ large ---
    "lunetpp_large_24f": LUNetPP_large_24f,
    "lunetpp_large_12f": LUNetPP_large_12f,
    "lunetpp_large_6f":  LUNetPP_large_6f,
}


def build_model(cfg: dict) -> UNet | MedNeXt | UNetPlusPlus:
    """Factory: instantiate a model preset from a config dict."""
    mcfg = cfg["model"]
    arch = mcfg["architecture"]
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[arch](
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        drop_prob=mcfg.get("drop_prob", 0.0),
    )
