from .c3d import c3d_model
from .r3d import r3d_model
from .r2plus1d import r2plus1d_model

__model_factory = {
    # image classification models
    "c3d": c3d_model,
    "r3d": r3d_model,
    "r2plus1d": r2plus1d_model,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)


