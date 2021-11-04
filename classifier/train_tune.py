from __future__ import print_function
from __future__ import division
import torch
import time
import os
import copy
import torchvision

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

TB_LOG_PATH = "./runs_Particulate_NonParticulate/"
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./Particulate_NonParticulate_classification_data"
model_save_path = "./Particulate_NonParticulate_weights"
create_dir(TB_LOG_PATH)
create_dir(data_dir)
create_dir(model_save_path)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"

# Number of classes in the dataset
num_classes = 2

is_inception = False

# Batch size for training (change depending on how much memory you have)
# batch_size = 8

# Number of epochs to train for
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
# feature_extract = False

# balanced_classes = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

writer = SummaryWriter(TB_LOG_PATH)

TRAIN = True

def make_weights_for_balanced_classes(images, nclasses):
    count = [0 for i in range(0, nclasses)]
    for i, item in enumerate(images):
        count[item[1]] += 1
    weight_per_class = [0. for i in range(0, nclasses)]
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0 for i in range(0, len(images))]
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def visualize_model(model, dataloaders, phase):
  confusion_matrix = torch.zeros(num_classes, num_classes)
  with torch.no_grad():
      for i, (inputs, classes) in enumerate(dataloaders[phase]):
          inputs = inputs.to(device)
          classes = classes.to(device)
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          for t, p in zip(classes.view(-1), preds.view(-1)):
                  confusion_matrix[t.long(), p.long()] += 1

#   print(confusion_matrix, '\n')
  particulate_f1 = 0.0
  particulate_precision = 0.0
  particulate_recall = 0.0
  new_matrix = confusion_matrix.tolist()
  pn_dict = [{'TP': 0.0, 'FP': 0.0, 'FN': 0.0} for i in range(num_classes)]
  for i in range(num_classes):
    for j in range(num_classes):
      if i == j:
        pn_dict[i]['TP'] += new_matrix[i][j]
      else:
        pn_dict[i]['FN'] += new_matrix[i][j]
        pn_dict[j]['FP'] += new_matrix[i][j]

  sum_f1 = 0.0
  for i in range(len(pn_dict)):
    TP = pn_dict[i]['TP']
    FP = pn_dict[i]['FP']
    FN = pn_dict[i]['FN']
    if (TP+FP) == 0.0:
        pre = 0.0
    else:
        pre = TP / (TP+FP)
    if (TP+FN) == 0.0:
        rec = 0.0
    else:
        rec = TP / (TP+FN)
    if (pre+rec) == 0.0:
        F1 = 0.0
    else:
        F1 = 2 * (pre * rec) / (pre + rec)
    # print('class: ', i, ', ', 'Precision = ', pre, ' Recall = ', rec, ' F1 score = ', F1)
    sum_f1 += F1
    if i == 1:
        particulate_f1 = F1
        particulate_precision = pre
        particulate_recall = rec

  avg_f1 = sum_f1 / num_classes
  print('avg. f1 = ', avg_f1)
  return particulate_f1, particulate_precision, particulate_recall

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name.startswith("resnet"):
        """ Resnet18
        """
        resnets = {
            "resnet18": models.resnet18(pretrained=use_pretrained),
            "resnet34": models.resnet34(pretrained=use_pretrained),
            "resnet50": models.resnet50(pretrained=use_pretrained)
        }
        model_ft = resnets[model_name]
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# config= {
#  model_name, feature_extract, randomhorizontalflip, balanced_classes, lr, batch_size}

# def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    

def test(config):
    model_ft, input_size = initialize_model(config["model_name"], num_classes, config["feature_extract"], use_pretrained=True)
    # Print the model we just instantiated
    print(model_ft)

    data_transforms = {
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}

    # Create training and validation dataloaders
    dataloaders_dict = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)}

    # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(os.path.join(model_save_path, "weights_resnet-18_resize.pt")))
    model_ft.eval()
    visualize_model(model_ft, dataloaders_dict, 'test')

def train(config, checkpoint_dir=None, data_dir=None):
    print(os.chdir("/home/asubramanian/TEM-XRD-pipeline/classifier"))
    # Initialize the model for this run
    model_ft, input_size = initialize_model(config["model_name"], num_classes, config["feature_extract"], use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    if config["flip"]:
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                # transforms.RandomResizedCrop((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                # transforms.RandomResizedCrop((input_size, input_size)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    weights_train = make_weights_for_balanced_classes(image_datasets['train'], len(image_datasets['train'].classes))
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))

    if not config["balanced_classes"]:
        sampler_train = None

    # Create training and validation dataloaders
    print(os.getcwd())
    dataloaders_dict = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=config["batch_size"], sampler = sampler_train, num_workers=4),
                        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=config["batch_size"], shuffle=False, num_workers=4)}

    # # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # model_ft = nn.DataParallel(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    # if TRAIN:
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if config["feature_extract"]:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    # else:
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=config["lr"], momentum=0.9)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_ft.load_state_dict(model_state)
        optimizer_ft.load_state_dict(optimizer_state)


    # Setup the loss fxn
    # criterion = {'train': nn.CrossEntropyLoss(weight=class_weights.to(device)), 'val': nn.CrossEntropyLoss()}
    criterion = {'train': nn.CrossEntropyLoss(), 'val': nn.CrossEntropyLoss()}

    # # Train and evaluate
    # model_ft, acc_hist, f1_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    # visualize_model(model_ft, dataloaders_dict, 'train')
    # visualize_model(model_ft, dataloaders_dict, 'val')

    since = time.time()

    # val_acc_history = []
    # val_f1_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train')
                print('-' * 10)
                model_ft.train()  # Set model to training mode
            else:
                print('Val')
                print('-' * 10)
                model_ft.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model_ft(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model_ft(inputs)
                        loss = criterion['val'](outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            f1, precision, recall = visualize_model(model_ft, dataloaders_dict, phase)

            # if phase == 'train':
            #     writer.add_scalar("train_loss", epoch_loss, epoch)
            #     writer.add_scalar("train_acc", epoch_acc, epoch)
            #     writer.add_scalar("train_F1", avg_f1, epoch)
            # elif phase == 'val':
            #     writer.add_scalar("val_loss", epoch_loss, epoch)
            #     writer.add_scalar("val_acc", epoch_acc, epoch)
            #     writer.add_scalar("val_F1", avg_f1, epoch)

            # # deep copy the model
            # if phase == 'val' and avg_f1 > best_f1:
            #     best_f1 = avg_f1
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)
            #     val_f1_history.append(avg_f1)
            # if phase == 'val':
            #   print("best val f1 ", best_f1)

            if phase == 'val':
                val_loss = epoch_loss
                val_acc = epoch_acc
                val_f1 = f1
                val_precision = precision
                val_recall = recall

            print("\n")

        print()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            # torch.save(model_ft.state_dict(), os.path.join(model_save_path, "weights_resnet-18_resize.pt"))
            torch.save((model_ft.state_dict(), optimizer_ft.state_dict()), path)

        tune.report(loss=val_loss, accuracy=val_acc, F1_score=val_f1, precision=val_precision, recall=val_recall)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))

    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    return model_ft, val_acc_history, val_f1_history
    # torch.save(model_ft.state_dict(), os.path.join(model_save_path, "weights_resnet-18_resize.pt"))
    
def main(num_samples=50, max_num_epochs=50, gpus_per_trial=2):

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64]),
        "model_name": tune.choice(['resnet18', "resnet34", "resnet50"]),
        "balanced_classes": tune.choice([True, False]),
        "feature_extract": tune.choice([True, False]),
        "flip": tune.choice([True, False])
    }
    scheduler = ASHAScheduler(
        metric="F1_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["training_iteration", "loss", "accuracy", "F1_score", "precision", "recall"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("F1_score", "max", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation F1: {}".format(
        best_trial.last_result["F1_score"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation precision: {}".format(
        best_trial.last_result["precision"]))
    print("Best trial final validation recall: {}".format(
        best_trial.last_result["recall"]))

    best_trained_model, _ = initialize_model(best_trial.config["model_name"], num_classes, best_trial.config["feature_extract"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     # device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    print("Best Checkpoint Directory", best_checkpoint_dir)
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()