import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_supported_model_modules_import():
    module_names = [
        "tractseg.models.unet_pytorch_deepsup",
        "tractseg.models.unet3d_pytorch_deepsup_sm",
        "tractseg.models.swinunetr",
        "tractseg.models.mednext",
        "tractseg.models.segment_anything",
        "tractseg.models.masam",
    ]

    for module_name in module_names:
        importlib.import_module(module_name)


def test_mednext_factory_is_available():
    from tractseg.models.mednext import create_mednext_v1

    assert callable(create_mednext_v1)


def test_sam_registry_is_available_from_integrated_package():
    from tractseg.models.segment_anything import sam_model_registry

    assert {"vit_b", "vit_l", "vit_h"}.issubset(sam_model_registry)


if __name__ == "__main__":
    test_supported_model_modules_import()
    test_mednext_factory_is_available()
    test_sam_registry_is_available_from_integrated_package()
    print("model import smoke ok")
