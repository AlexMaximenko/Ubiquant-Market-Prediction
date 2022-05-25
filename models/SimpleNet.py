import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class LightningSimpleNet(LightningModule):
    def __init__(self, features_num=300, embed_size=32, investment_id_num=4100, investment_final_dim = 64, hidden_size=256, features_final_dim=256):
        super().__init__()
        self.investment_net = nn.Sequential(
            nn.Embedding(investment_id_num, embed_size),
            nn.Linear(embed_size, embed_size * 2),
            nn.BatchNorm1d(embed_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_size * 2, investment_final_dim),
            nn.BatchNorm1d(investment_final_dim),
            nn.ReLU()
        )

        self.feature_net = nn.Sequential(
            nn.Linear(features_num, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, features_final_dim),
            nn.BatchNorm1d(features_final_dim),
            nn.ReLU()

        )

        self.final_net = nn.Sequential(
            nn.Linear(investment_final_dim + features_final_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        investment_output = self.investment_net(x['investment_ids'])
        features_output = self.feature_net(x['features'])
        concated = torch.cat((investment_output, features_output), dim=1)
        output = self.final_net(concated)
        return output

    def training_step(self, batch, batch_idx):
        loss = F.mse_loss(self.forward(batch), batch['targets'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = F.mse_loss(self.forward(batch), batch['targets'])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx=0, dataloader_idx=None):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer}
    


