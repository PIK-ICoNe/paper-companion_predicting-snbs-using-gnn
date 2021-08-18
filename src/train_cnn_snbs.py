import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from cnn_models import cnn_snbs
# from myResnet import resnet18
import torchvision.models as models


print("import finished")


cfg = {}
cfg["manual_seed"] = 1
cfg["dataset::path"] = "../../4Pytorch"
cfg["train_set::batchsize"] = 100
cfg["test_set::batchsize"] = 200
cfg["train_set::start_index"] = 0
cfg["train_set::end_index"] = 799
cfg["test_set::start_index"] = 800
cfg["test_set::end_index"] = 999
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["epochs"] = 100
cfg["model"] = "resnet50"
cfg["optim::optimizer"] = "SGD"
cfg["optim::LR"] = .3
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["cfg_path"] = "./"
cfg["criterion"] = "MSELoss"
cfg["eval::threshold"] = .1
cfg["model::num_input_features"] = 1
cfg["model::num_classes"] = 20


json_config = json.dumps(cfg)
f = open(cfg["cfg_path"] + "training_cfg.json", "w")
f.write(json_config)
f.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting the seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
np.random.seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_set = cnn_snbs(cfg["dataset::path"], slice_index=slice(
    cfg["train_set::start_index"], cfg["train_set::end_index"]), num_input_features=cfg["model::num_input_features"])
test_set = cnn_snbs(cfg["dataset::path"], slice_index=slice(
    cfg["test_set::start_index"], cfg["test_set::end_index"]),  num_input_features=cfg["model::num_input_features"])


if cfg["model"] == "resnet18":
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(cfg["model::num_input_features"],
                            64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, cfg["model::num_classes"])


if cfg["model"] == "resnet34":
    model = models.resnet34(pretrained=True)
    model.conv1 = nn.Conv2d(cfg["model::num_input_features"],
                            64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, cfg["model::num_classes"])

if cfg["model"] == "resnet50":
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(cfg["model::num_input_features"],
                            64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, cfg["model::num_classes"])

model.double()


if cfg["criterion"] == "MSELoss":
    criterion = nn.MSELoss()
if cfg["optim::optimizer"] == "SGD":
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg["optim::LR"], momentum=cfg["optim::momentum"])
if cfg["optim::optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(
    ), lr=cfg["optim::LR"], weight_decay=cfg["optim::weight_decay"])


train_loader = DataLoader(
    train_set, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"])
test_loader = DataLoader(
    test_set, batch_size=cfg["test_set::batchsize"], shuffle=cfg["test_set::shuffle"])


def train_epoch(model, data_loader):
    loss = 0.0
    for iter, (batch) in enumerate(data_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        # batch.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = torch.sigmoid(model.forward(inputs))
        temp_loss = criterion(outputs, labels)
        temp_loss.backward()
        optimizer.step()
        loss += temp_loss.item()
        print('[' + '{:5}'.format(iter * cfg["train_set::batchsize"]) + '/' + '{:5}'.format(len(train_set)) +
              ' (' + '{:3.0f}'.format(100 * iter / len(train_loader)) + '%)] Train Loss: ' +
              '{:6.4f}'.format(temp_loss.item()))
    return loss


def eval(model, data_loader, tol):
    model.eval()
    torch.no_grad()
    N = data_loader.dataset[0][1].shape[0]
    loss = 0.
    correct = 0
    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        output = torch.sigmoid(model(inputs))
        temp_loss = criterion(output, labels)
        loss += temp_loss.item()
        correct += get_prediction(output, labels, tol)
    accuracy = 100 * correct / (N*len(test_loader)*test_loader.batch_size)
    print(f"Test loss: {loss/len(data_loader):.3f}.. "
          f"Test accuracy: {accuracy:.3f} %"
          )
    return loss, accuracy


def get_prediction(output, label, tol):
    count = 0
    output = output.view(-1, 1).view(-1)
    label = label.view(-1, 1).view(-1)
    batchSize = output.size(-1)
    for i in range(batchSize):
        if ((abs(output[i] - label[i]) < tol).item() == True):
            count += 1
    return count


def compute_R2(model, data_loader):
    # R**2 = 1 - mse(y,t) / mse(t_mean,t)
    N = data_loader.dataset[0][1].shape[0]
    all_labels = torch.empty(0, N)
    all_labels.to(device)
    model.eval()
    torch.no_grad()
    mse_trained = 0
    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        output = torch.sigmoid(model(inputs))
        mse_trained += torch.sum((output - labels) ** 2)
        all_labels = torch.cat((all_labels.double(), labels), dim=0)
    mean_labels = torch.mean(all_labels)
    output_mean = mean_labels * torch.ones(len(all_labels), N)
    mse_mean = torch.sum((output_mean-all_labels)**2)
    return (1 - mse_trained/mse_mean).item()


model.to(device)
model.double()


train_loss, test_loss = [], []
test_accuracy = []
R2_score = []
epochs = cfg["epochs"]
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}.. ")
    temp_loss = train_epoch(model, train_loader)
    train_loss.append(temp_loss)
    temp_test_loss, test_accu = eval(
        model, test_loader, cfg["eval::threshold"])
    test_loss.append(temp_test_loss)
    test_accuracy.append(test_accu)
    R2 = compute_R2(model, test_loader)
    if len(R2_score) > 1:
        if R2 > max(R2_score):
            torch.save(model.state_dict(), cfg["cfg_path"] + "best_model.pt")
    print('R2: ''{:3.2f}'.format(100 * R2) + '%')
    R2_score.append(R2)

best_accuracy_index = test_accuracy.index(max(test_accuracy))
best_R2_index = R2_score.index(max(R2_score))
print("Epoch of best test_accuracy: ", best_accuracy_index+1,
      "  Accuracy: ", test_accuracy[best_accuracy_index], '%')
print("Epoch of best R2_score: ", best_R2_index,
      '   R2: ''{:3.2f}'.format(100 * R2_score[best_R2_index]) + '%')


training_results = {}
training_results["train_loss"] = train_loss
training_results["test_loss"] = test_loss
training_results["test_accuracy"] = test_accuracy
training_results["R2_score"] = R2_score


json_results = json.dumps(training_results)
f = open(cfg["cfg_path"] + "training_results.json", "w")
f.write(json_results)
f.close()
print("results of training are stored in " + cfg["cfg_path"])
