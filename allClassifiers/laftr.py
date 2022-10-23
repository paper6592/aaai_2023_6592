import torch
from torch import nn
import torch.nn.functional as F

# Taken From: https://github.com/windxrz/DCFR

def weighted_cross_entropy(w, y, y_pred, eps=1e-8):
    res = -torch.sum(
        w * (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
    )
    return res

def weighted_mae(w, y, y_pred):
    return torch.sum(w * torch.abs(y - y_pred))

def weighted_mse(w, y, y_pred):
    return torch.sum(w * (y - y_pred) * (y - y_pred))

class MLP(nn.Module):
    def __init__(self, shapes, acti):
        super(MLP, self).__init__()
        self.acti = acti
        self.fc = nn.ModuleList()
        for i in range(0, len(shapes) - 1):
            self.fc.append(nn.Linear(shapes[i], shapes[i + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i == len(self.fc) - 1:
                break
            if self.acti == "relu":
                x = F.relu(x)
            elif self.acti == "sigmoid":
                x = F.sigmoid(x)
            elif self.acti == "softplus":
                x = F.softplus(x)
            elif self.acti == "leakyrelu":
                x = F.leaky_relu(x)
        return x

    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def activate(self):
        for para in self.parameters():
            para.requires_grad = True

class FairRepr(nn.Module):
    def __init__(self, model_name, config):
        super(FairRepr, self).__init__()
        dataset = config["dataset"]
        task = config["task"]
        self.fair_coeff = config["fair_coeff"]
        self.task = task
        self.name = f"{model_name}_{task}_{dataset}_fair_coeff_{self.fair_coeff}"

    def loss_prediction(self, x, y, w):
        return 0

    def loss_audit(self, x, s, f, w):
        return 0

    def loss(self, x, y, s, f, w_pred, w_audit):
        loss = self.loss_prediction(x, y, w_pred) - self.fair_coeff * self.loss_audit(
            x, s, f, w_audit
        )
        return loss

    def weight_pred(self, df):
        n = df.shape[0]
        return torch.ones((n, 1)) / n

    def weight_audit(self, df, s, f):
        return torch.tensor([1.0 / df.shape[0]] * df.shape[0])

    def forward_y(self, x):
        pass

    def forward(self, x):
        self.forward_y(x)

def train_laftr(train_dataset):
    config = {
        "xdim": train_dataset.features[:,]
    }
    return None

class LAFTR(FairRepr):
    def __init__(self, config):
        super(LAFTR, self).__init__("LAFTR", config)
        self.encoder = MLP(
            [config["xdim"]] + config["encoder"] + [config["zdim"]], "relu"
        )
        self.prediction = MLP(
            [config["zdim"]] + config["prediction"] + [config["ydim"]],
            "relu",
        )
        self.audit = MLP([config["zdim"]] + config["audit"] + [config["sdim"]], "relu")

    def forward_y(self, x):
        z = self.forward_z(x)
        y = self.prediction(z)
        y = torch.sigmoid(y)
        return y

    def forward_s(self, x, f):
        z = self.forward_z(x)
        s = self.audit(z)
        s = torch.sigmoid(s)
        return s

    def forward_z(self, x):
        z = torch.nn.functional.relu(self.encoder(x))
        return z

    def forward(self, x):
        self.forward_y(x)

    def loss_prediction(self, x, y, w):
        y_pred = self.forward_y(x)
        loss = weighted_cross_entropy(w, y, y_pred)
        return loss

    def loss_audit(self, x, s, f, w):
        s_pred = self.forward_s(x, f)
        loss = weighted_mse(w, s, s_pred)
        return loss

    def weight_audit(self, df_old, s, f):
        df = df_old.copy()
        df["w"] = 0.0
        if self.task == "DP":
            amount = df.loc[df[s] == 1].shape[0]
            df.loc[df[s] == 1, "w"] = 1.0 / amount / 2
            df.loc[df[s] == 0, "w"] = 1.0 / (df.shape[0] - amount) / 2
        elif self.task == "EO":
            for ss in range(2):
                for y in range(2):
                    amount = df.loc[(df[s] == ss) & (df["result"] == y)].shape[0]
                    df.loc[(df[s] == ss) & (df["result"] == y), "w"] = 1.0 / amount / 4
        elif self.task == "CF":
            res = (
                df.groupby(f + [s])
                .count()["w"]
                .reset_index()
                .rename(columns={"w": "n_s_f"})
            )
            df = df.merge(res, on=f + [s], how="left")
            df["w"] = 1.0 / df["n_s_f"]

        res = torch.from_numpy(df["w"].values).view(-1, 1)
        res = res / res.sum()
        return res

    def predict_only(self):
        self.audit.freeze()
        self.prediction.activate()
        self.encoder.activate()

    def audit_only(self):
        self.audit.activate()
        self.prediction.freeze()
        self.encoder.freeze()

    def finetune_only(self):
        self.audit.freeze()
        self.prediction.activate()
        self.encoder.freeze()

    def predict_params(self):
        return list(self.prediction.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        return self.audit.parameters()

    def finetune_params(self):
        return self.prediction.parameters()