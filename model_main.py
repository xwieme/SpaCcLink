import os
import pandas as pd
import numpy as np
import torch
import datetime
from utils.utils_train import load_data, get_cosine_schedule_with_warmup
from utils.utils import getData, switchId
from model.model import GraphModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import configparser
from sklearn.metrics import f1_score


def fit_one_epoch(
    model,
    train_loader,
    test_loader,
    data,
    optimizer,
    epoch,
    Epoch,
    n_train_batch,
    n_test_batch,
    device,
    loss_file,
    save_period=5,
):

    x = data.x.to(device)
    adj = data.edges.to(device)

    train_loss = 0
    test_loss = 0

    # -------------------- train -------------------------------
    model.train()
    for batch in train_loader:

        pairs = batch.to(device)

        optimizer.zero_grad()
        loss = model.computeLoss(x, adj, pairs[:, 0], pairs[:, 1], pairs[:, 2])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / n_train_batch

    # -------------------- test -------------------------------
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            pairs = batch.to(device)

            loss = model.computeLoss(x, adj, pairs[:, 0], pairs[:, 1], pairs[:, 2])

            test_loss += loss.item()

    test_loss = test_loss / n_test_batch

    print(
        "Epoch:"
        + str(epoch + 1)
        + "/"
        + str(Epoch)
        + "\t\t"
        + "Train Loss: %.3f  Test Loss: %.3f " % (train_loss, test_loss)
    )

    with open(loss_file, "a") as f:
        f.write(str(train_loss) + " " + str(test_loss) + "\n")

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), "logs/epoch-%03d.pth" % (epoch + 1))


def train(config=None):
    if config is None:
        cp = configparser.ConfigParser()
        cp.read("config.cfg")
    else:
        cp = config
    num_workers = cp.getint("loader", "num_workers")
    batch_size = cp.getint("loader", "batch_size")

    Epoch = cp.getint("train", "Epoch")
    save_period = cp.getint("train", "save_period")
    lr = cp.getfloat("train", "lr")

    out_dim = cp.getint("model", "out_dim")
    num_head = cp.getint("model", "num_head")

    input_dir = cp.get("data", "input_dir")

    train_data, test_data, express_matrix, pathways = load_data(input_dir)

    mapping = dict(
        zip(express_matrix.columns.values, range(len(express_matrix.columns.values)))
    )

    data = getData(express_matrix, pathways, mapping)

    input_dim = data.x.shape[1]

    if not os.path.exists("logs"):
        os.makedirs("logs")
    torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphModel(input_dim, out_dim, num_head=num_head).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, Epoch * 0.1, Epoch)

    train_dataset = switchId(train_data, mapping)
    test_dataset = switchId(test_data, mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
    )

    n_train_batch = len(train_dataset) // batch_size
    n_test_batch = len(test_dataset) // batch_size

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")

    loss_file = "logs/loss_" + str(time_str) + ".txt"

    for epoch in tqdm(range(Epoch)):

        fit_one_epoch(
            model,
            train_loader,
            test_loader,
            data,
            optimizer,
            epoch,
            Epoch,
            n_train_batch,
            n_test_batch,
            device,
            loss_file,
            save_period,
        )

        scheduler.step()


def test(config=None):

    if config is None:
        cp = configparser.ConfigParser()
        cp.read("config.cfg")
    else:
        cp = config

    num_workers = cp.getint("loader", "num_workers")
    batch_size = cp.getint("loader", "batch_size")

    out_dim = cp.getint("model", "out_dim")
    num_head = cp.getint("model", "num_head")
    thre = cp.getfloat("model", "thre")

    model_path = cp.get("test", "model_path")

    input_dir = cp.get("data", "input_dir")

    test_data, express_matrix, pathways = load_data(input_dir, "test")

    mapping = dict(
        zip(express_matrix.columns.values, range(len(express_matrix.columns.values)))
    )

    data = getData(express_matrix, pathways, mapping)

    input_dim = data.x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphModel(input_dim, out_dim, num_head=num_head).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_dataset = switchId(test_data, mapping)
    pos = test_dataset[:, :2]

    pos = torch.cat((pos, torch.ones(pos.shape[0], dtype=torch.int).view(-1, 1)), dim=1)
    neg = test_dataset[:, [0, 2]]
    neg = torch.cat(
        (neg, torch.zeros(neg.shape[0], dtype=torch.int).view(-1, 1)), dim=1
    )
    test_dataset = torch.cat((pos, neg))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    x = data.x.to(device)
    adj = data.edges.to(device)

    true_label = []
    pred_label = []

    with torch.no_grad():
        for batch in test_loader:
            pairs = batch[:, :2].to(device)

            pred, _ = model.inference(x, adj, pairs, thre_cos=thre)

            true_label += batch[:, 2].tolist()

            pred_label += pred.tolist()

    f1 = f1_score(true_label, pred_label)
    return f1


def predict():
    cp = configparser.ConfigParser()
    cp.read("config.cfg")

    num_workers = cp.getint("loader", "num_workers")
    batch_size = cp.getint("loader", "batch_size")

    out_dim = cp.getint("model", "out_dim")
    num_head = cp.getint("model", "num_head")
    thre = cp.getfloat("model", "thre")

    model_path = cp.get("test", "model_path")

    input_dir = cp.get("data", "input_dir")

    pred_data, express_matrix, pathways = load_data(input_dir, "pred")

    mapping = dict(
        zip(express_matrix.columns.values, range(len(express_matrix.columns.values)))
    )

    data = getData(express_matrix, pathways, mapping)

    input_dim = data.x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphModel(input_dim, out_dim, num_head=num_head).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pred_dataset = switchId(pred_data, mapping)

    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    x = data.x.to(device)
    adj = data.edges.to(device)

    pred_label = []
    relation = []
    with torch.no_grad():
        for batch in pred_loader:
            pairs = batch.to(device)

            pred, simlarity = model.inference(x, adj, pairs, thre_cos=thre)
            pred_label += pred.tolist()
            relation += simlarity.tolist()

    top_pairs_idx = np.array(pred_label) == 1

    top_pairs = pred_data.loc[top_pairs_idx]
    top_pairs_sim = np.array(relation)[top_pairs_idx]
    top_pairs["relation"] = top_pairs_sim

    top_pairs.to_csv(
        os.path.join(cp.get("data", "output_dir"), "top_pairs.csv"), index=False
    )


if __name__ == "__main__":
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train()
