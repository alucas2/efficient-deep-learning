import torch
import torchinfo
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from data import *
import trainer2
import trainer3
import trainer4
from utils import *
import numpy as np
#import torch.quantization.quantize_fx as quantize_fx
import copy

USE_CIFAR10 = True
USE_THIN_RESNET18 = False
USE_RESNET20 = False
USE_GROUP_RESNET18 = False
USE_GROUP_RESNET20 = True 
USE_HALF = False
USE_MIXUP = True
USE_AUTOAUGMENTED = False
USE_DISTILLATION = False
BATCH_SIZE = 32
USE_BINARIZATION = False
USE_QUANTIZATION_AWARE = False

dataset_description = []

if USE_AUTOAUGMENTED:
    transform_train = TRANSFORM_TRAIN_AUTOAUGMENTED
    transform_test = TRANSFORM_TEST
    dataset_description.append("autoaugmented")
else:
    transform_train = TRANSFORM_TRAIN
    transform_test = TRANSFORM_TEST

# Load the datasets
if USE_CIFAR10:
    train_dataset = get_cifar10_train(transform_train)
    test_dataset = get_cifar10_test(transform_test)
    num_classes = 10
    dataset_description.append("cifar10")
else:
    train_dataset = get_minicifar_train(transform_train)
    test_dataset = get_minicifar_test(transform_test)
    num_classes = 4
    dataset_description.append("minicifar")

# Create the model
if USE_THIN_RESNET18:
    model_name = "thinresnet18"
    model = make_thinresnet18(num_classes)
elif USE_RESNET20:
    model_name = "resnet20"
    model = make_resnet20(num_classes)
elif USE_GROUP_RESNET18:
    model_name = "group_resnet18"
    model = make_group_resnet18(num_classes)
elif USE_GROUP_RESNET20:
    model_name = "group8_resnet20"
    model = make_group_resnet20(num_classes)
else:
    model_name = "normalresnet18"
    model = make_resnet18(num_classes)

# --------------------------------------------------------------------------------------------------------

train_preprocess = []
test_preprocess = []
loss = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    train_preprocess.append(batch_to_cuda)
    test_preprocess.append(batch_to_cuda)

if USE_HALF:
    dataset_description.append("half")
    model = model.half()
    train_preprocess.append(batch_to_half)
    test_preprocess.append(batch_to_half)

if USE_MIXUP: # the train accuracy is garbage when mixup is activated
    dataset_description.append("mixup")
    train_preprocess.append(batch_mixup)

if USE_DISTILLATION:
    dataset_description.append("distil")
    teacher_model = make_resnet20(num_classes)
    teacher_model.load_state_dict(torch.load("models/resnet20_for_cifar10_mixup.pth"))
    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()
    loss = DistillationLoss(teacher_model, 0.5)

if USE_BINARIZATION: # the train accuracy is garbage when mixup is activated
    model_name += '_binarized'

if USE_QUANTIZATION_AWARE:
    model_name += '_quantized'

# --------------------------------------------------------------------------------------------------------

file_name = model_name + "_for_" + '_'.join(dataset_description)
print(file_name)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training settings
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# Train the model
if USE_BINARIZATION: 
    trainer = trainer3.Trainer(model, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)
    metrics, best_model = trainer.train(num_epochs=150)

#if USE_QUANTIZATION_AWARE:
    # model.load_state_dict(torch.load("/homes/r20benta/Documents/E-DEEP/potential-tribble/models/normalresnet18_for_cifar10_mixup.pth"))
    # model_to_quantize = copy.deepcopy(model)
    # qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
    # model_to_quantize.train()
    # prepare
    #model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict)
    # training loop (not shown)
    # quantize
    #model_quantized = quantize_fx.convert_fx(model_prepared)

    #
    # fusion
    #
    # model_to_quantize = copy.deepcopy(model)
    # model_fused = quantize_fx.fuse_fx(model_to_quantize)
    # trainer = trainer2.Trainer(model_fused, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)
    # metrics, best_model = trainer.train(num_epochs=150)
    # fused_model= trainer4.model_fusion(model)
    # quantized_model = QuantizedResNet18(model_fp32=fused_model)
    # quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # quantized_model.qconfig = quantization_config
    # torch.quantization.prepare_qat(quantized_model, inplace=True)
    # trainer = trainer4.Trainer(quantized_model, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)
    # metrics, best_model = trainer.train(num_epochs=150)

else:
    trainer = trainer2.Trainer(model, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)
    metrics, best_model = trainer.train(num_epochs=150)

# Save the model and metrics
metrics.save(f"logs/{file_name}.csv")
torch.save(best_model.state_dict(), f"models/{file_name}.pth")

# Save the model summary
with open(f"models/{file_name}.txt", "w") as f:
    f.write(str(torchinfo.summary(
        model, input_size=(BATCH_SIZE, 3, 32, 32), dtypes=[torch.half] if USE_HALF else [torch.float]
    )))
