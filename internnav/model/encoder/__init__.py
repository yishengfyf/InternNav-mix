from importlib import import_module


_LAZY_IMPORTS = {
    "PositionalEncoding": (".bert_backbone", "PositionalEncoding"),
    "DistanceNetwork": (".distance_encoder", "DistanceNetwork"),
    "ImageEncoder": (".image_clip_encoder", "ImageEncoder"),
    "InstructionEncoder": (".instruction_encoder", "InstructionEncoder"),
    "InstructionLongCLIPEncoder": (".instruction_longCLIP_encoder", "InstructionLongCLIPEncoder"),
    "LanguageEncoder": (".instruction_roberta_encoder", "LanguageEncoder"),
    "VisionLanguageEncoder": (".vision_language_encoder", "VisionLanguageEncoder"),
    "resnet_encoders": (".resnet_encoders", None),
}

__all__ = tuple(_LAZY_IMPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
