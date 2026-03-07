from .swin_v1 import SwinTransformer, swin_v1_l


def build_backbone() -> SwinTransformer:
    return swin_v1_l()
