from __future__ import annotations

import torch

from vjepa_forge.models import predictor as vit_predictor
from vjepa_forge.models import vision_transformer as vit_encoder


ARCH_NAME_MAP = {
    "vjepa2_1_vit_base_384": ("vit_base", "vjepa2_1_vitb_dist_vitG_384"),
    "vjepa2_1_vit_large_384": ("vit_large", "vjepa2_1_vitl_dist_vitG_384"),
    "vjepa2_1_vit_giant_384": ("vit_giant_xformers", "vjepa2_1_vitg_384"),
    "vjepa2_1_vit_gigantic_384": ("vit_gigantic_xformers", "vjepa2_1_vitG_384"),
}

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
VJEPA21_TEACHER_EMBED_DIM = 1664


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def _make_vjepa2_1_model(
    *,
    model_name: str,
    checkpoint_key: str = "ema_encoder",
    img_size: int = 384,
    patch_size: int = 16,
    tubelet_size: int = 2,
    num_frames: int = 64,
    predictor_embed_dim: int = 384,
    predictor_depth: int = 24,
    predictor_num_mask_tokens: int = 10,
    n_output_distillation: int = 4,
    return_all_tokens: bool = False,
    teacher_embed_dim: int | None = None,
    pretrained: bool = True,
    **kwargs,
):
    vit_encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    vit_encoder_kwargs.update(**kwargs)
    arch_name = ARCH_NAME_MAP[model_name][0]
    encoder = vit_encoder.__dict__[arch_name](**vit_encoder_kwargs)

    vit_predictor_kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        teacher_embed_dim=teacher_embed_dim,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        depth=predictor_depth,
        num_heads=12,
        num_mask_tokens=predictor_num_mask_tokens,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        n_output_distillation=n_output_distillation,
        return_all_tokens=return_all_tokens,
        img_temporal_dim_size=1,
    )
    vit_predictor_kwargs.update(**kwargs)
    predictor = vit_predictor.__dict__["vit_predictor"](**vit_predictor_kwargs)

    if pretrained:
        model_file = ARCH_NAME_MAP[model_name][-1]
        state_dict = torch.hub.load_state_dict_from_url(f"{VJEPA_BASE_URL}/{model_file}.pt", map_location="cpu")
        encoder.load_state_dict(_clean_backbone_key(state_dict[checkpoint_key]), strict=True)
        predictor.load_state_dict(_clean_backbone_key(state_dict["predictor"]), strict=True)

    return encoder, predictor


def vjepa2_1_vit_base_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_base_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=VJEPA21_TEACHER_EMBED_DIM,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_large_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_large_384",
        checkpoint_key="ema_encoder",
        img_size=384,
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=VJEPA21_TEACHER_EMBED_DIM,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_giant_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_giant_384",
        img_size=384,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        pretrained=pretrained,
        **kwargs,
    )


def vjepa2_1_vit_gigantic_384(*, pretrained: bool = True, **kwargs):
    return _make_vjepa2_1_model(
        model_name="vjepa2_1_vit_gigantic_384",
        img_size=384,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        pretrained=pretrained,
        **kwargs,
    )
