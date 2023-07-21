from .c3d import c3d_model
from .r3d import r3d_18_model, r3d_34_model, r2plus1d_18_model, mcs_18_model, r2d_18_model
from .r2plus1d import r2p1d_18_model, r2p1d_34_model
from .vivit import vivit_model


__model_factory = {
    # image classification models
    "c3d": c3d_model,
    "r3d_18": r3d_18_model,
    "r3d_34": r3d_34_model,
    "r2p1d_18": r2p1d_18_model,
    "r2p1d_34": r2p1d_34_model,
    "vivit": vivit_model,
    "r2plus1d_18": r2plus1d_18_model,
    "mcs_18": mcs_18_model,
    "r2d_18": r2d_18_model,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)


