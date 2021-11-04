from classifier.evaluator import Evaluator
from classifier.factory import BuilderFactory


def sem_tem_other_builder(data_path, output_dir):
    weights_path = "classifier/SEM_TEM_Other_weights/weights_resnet-18_resize.pt"
    num_classes = 2
    id_to_label = {0: 'Other', 1: 'TEM'}
    return Evaluator(weights_path, num_classes, data_path, output_dir, id_to_label)

def tem_xrd_other_builder(data_path, output_dir):
    weights_path = "classifier/TEM_XRD_Other_weights/weights_resnet-18_resize.pt"
    num_classes = 3
    id_to_label = {0: 'Other', 1: 'TEM', 2: 'XRD'}
    return Evaluator(weights_path, num_classes, data_path, output_dir, id_to_label)

def tem_subcategories_builder(data_path, output_dir):
    weights_path = "classifier/Diffraction_Elemental_HRTEM_Normal_Other_weights/weights_resnet-18.pt"
    num_classes = 5
    id_to_label = {0: 'TEM_diffraction', 1: 'TEM_elemental', 2: 'TEM_hrtem', 3: 'TEM_normal', 4: 'TEM_other'}
    return Evaluator(weights_path, num_classes, data_path, output_dir, id_to_label)

def particulate_builder(data_path, output_dir):
    weights_path = "classifier/Particulate_nonParticulate_weights/weights_resnet-18_resize.pt"
    num_classes = 2
    id_to_label = {0: 'Non_particulate', 1: 'Particulate'}
    return Evaluator(weights_path, num_classes, data_path, output_dir, id_to_label)

factory = BuilderFactory()
factory.register_builder("tem_xrd_other", tem_xrd_other_builder)
factory.register_builder("tem_subcategories", tem_subcategories_builder)
factory.register_builder("particulate", particulate_builder)
factory.register_builder("sem_tem_other", sem_tem_other_builder)
