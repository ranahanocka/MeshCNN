def create_model(opt):
    from .mesh_classifier import ClassifierModel  # todo - get rid of this ?

    model = ClassifierModel(opt)
    return model
