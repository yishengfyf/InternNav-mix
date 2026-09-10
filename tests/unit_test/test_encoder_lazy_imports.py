import importlib
import sys


def test_depth_anything_import_does_not_require_longclip():
    encoder_package = importlib.import_module("internnav.model.encoder")

    assert "internnav.model.encoder.image_clip_encoder" not in sys.modules
    assert "ImageEncoder" not in vars(encoder_package)

    depth_module = importlib.import_module(
        "internnav.model.encoder.depth_anything.depth_anything_v2.dpt"
    )

    assert hasattr(depth_module, "DepthAnythingV2")
    assert "internnav.model.encoder.image_clip_encoder" not in sys.modules
