import os
import torch
import shutil
from torchvision import transforms
from classifier.model import Model
from classifier.dataset import ClassificationDataset


class Evaluator:
    # TODO(aksub99): Add option to test on GPU.

    def __init__(self, weights_path, num_classes, data_path, output_dir, id_to_label):
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.data_path = data_path
        self.output_dir = output_dir
        self.id_to_label = id_to_label

    def create_transforms(self, input_size):
        data_transforms = {
            'predict': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms

    def create_output_directories(self):
        for _, value in self.id_to_label.items():
            if not os.path.isdir(os.path.join(self.output_dir, value)):
                os.mkdir(os.path.join(self.output_dir, value))

    def prepare_model(self, device):
        model = Model(self.num_classes)
        pretrained_model, input_size = model.initialize_model()
        pretrained_model.to(device)

        if device == torch.device("cpu"):
            pretrained_model.load_state_dict(torch.load(self.weights_path, map_location='cpu'))
        else:
            pretrained_model.load_state_dict(torch.load(self.weights_path))
        pretrained_model.eval()
        return pretrained_model, input_size

    def prepare_dataloader(self, input_size):
        transforms = self.create_transforms(input_size)
        dataset = {'predict' : ClassificationDataset(self.data_path, transforms['predict'])}
        dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size=1, shuffle=False, num_workers=4)}
        return dataloader

    def infer(self, GPU):
        device = torch.device("cuda:0" if GPU else "cpu")
        self.create_output_directories()
        pretrained_model, input_size = self.prepare_model(device)
        dataloader = self.prepare_dataloader(input_size)

        outputs = list()
        for inputs, img_name in dataloader['predict']:
            output = pretrained_model(inputs.to(device))
            index = output.data.cpu().numpy().argmax()
            shutil.copyfile(os.path.join(self.data_path, img_name[0]),
                            os.path.join(self.output_dir, '{}/'.format(self.id_to_label[index]), img_name[0]))
