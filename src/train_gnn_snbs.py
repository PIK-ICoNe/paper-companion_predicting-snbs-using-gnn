import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import DataLoader

from gnn_models import gnn_snbs, initialize_model


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
cfg["epochs"] = 500
cfg["model"] = "ArmaNet02"
cfg["optim::optimizer"] = "SGD"
cfg["optim::LR"] = .3
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["cfg_path"] = "./"
cfg["criterion"] = "MSELoss"
cfg["eval::threshold"] = .1


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


train_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
    cfg["train_set::start_index"], cfg["train_set::end_index"]))
test_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
    cfg["test_set::start_index"], cfg["test_set::end_index"]))


model = initialize_model(cfg["model"])


if cfg["criterion"] == "MSELoss":
    criterion = nn.MSELoss()
criterion.to(device)

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
    model.train()
    for iter, (batch) in enumerate(data_loader):
        batch.to(device)
        optimizer.zero_grad()
        outputs = model.forward(batch)
        labels = batch.y
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
    with torch.no_grad():
        N = data_loader.dataset[0].x.shape[0]
        loss = 0.
        correct = 0
        all_labels = torch.Tensor(0).to(device)
        mse_trained = 0.
        for batch in data_loader:
            batch.to(device)
            labels = batch.y
            output = model(batch)
            mse_trained += torch.sum((output - labels) ** 2)
            temp_loss = criterion(output, labels)
            loss += temp_loss.item()
            correct += get_prediction(output, labels, tol)
            all_labels = torch.cat([all_labels, labels])
    accuracy = 100 * correct / (all_labels.shape[0])
    print(f"Test loss: {loss/len(data_loader):.3f}.. "
          f"Test accuracy: {accuracy:.3f} %"
          )
    mean_labels = torch.mean(all_labels)
    array_ones = torch.ones(all_labels.shape[0], 1)
    array_ones = array_ones.to(device)
    output_mean = mean_labels * array_ones
    mse_mean = torch.sum((output_mean-all_labels)**2)
    R2 = (1 - mse_trained/mse_mean).item()
    return loss, accuracy, R2


def get_prediction(output, label, tol):
    count = 0
    output = output.view(-1, 1).view(-1)
    label = label.view(-1, 1).view(-1)
    batchSize = output.size(-1)
    for i in range(batchSize):
        if ((abs(output[i] - label[i]) < tol).item() == True):
            count += 1
    return count


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
    temp_test_loss, test_accu, R2 = eval(
        model, test_loader, cfg["eval::threshold"])
    test_loss.append(temp_test_loss)
    test_accuracy.append(test_accu)
    if len(R2_score) > 1:
        if R2 > max(R2_score):
            torch.save(model.state_dict(), cfg["cfg_path"] + "best_model.pt")
    print('R2: ''{:3.2f}'.format(100 * R2) + '%')
    R2_score.append(R2)

best_accuracy_index = test_accuracy.index(max(test_accuracy))
best_R2_index = R2_score.index(max(R2_score))
print("Epoch of best test_accuracy: ", best_accuracy_index+1,
      "  Accuracy: ", test_accuracy[best_accuracy_index], '%')
print("Epoch of best R2_score: ", best_R2_index+1,
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
