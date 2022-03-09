import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class LightningMemoryNet(LightningModule):
    """
    This model uses LSTM + MLP for predictions. 
    For each 'investment_id' it saves last hidden_state and cell_state.
    """
    def __init__(
        self, 
        features_num=300, 
        hidden_size=256
        ):
        super().__init__()
        
        self.cell_states = {}
        self.hidden_states = {}
        self.LSTM = nn.LSTM(
            input_size=features_num,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.hidden_size = hidden_size
        self.features_num = features_num

        self.register_buffer("init_hidden_state", torch.autograd.Variable(torch.zeros([1, self.hidden_size]), requires_grad=True))
        self.register_buffer("init_cell_state", torch.autograd.Variable(torch.zeros([1, self.hidden_size]), requires_grad=True))
        
        #self.init_hidden_state = torch.autograd.Variable(torch.zeros([1, self.hidden_size]), requires_grad=True)
        #self.init_cell_state = torch.autograd.Variable(torch.zeros([1, self.hidden_size]), requires_grad=True)

        self.last_layer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        """
        Input:  
            x: list of:
                    x[0] = torch.tensor[Batch_size, Features_num] - features
                    x[1] = torch.tensor[Batch_size, 1] - investment_ids
        """
        hidden_states, cell_states = self._get_states(torch.squeeze(x[1]))
        features = x[0]

        output, (h_n, c_n) = self.LSTM(features.unsqueeze(1), [hidden_states.unsqueeze(0), cell_states.unsqueeze(0)])
        
        for i in range(output.shape[0]):
            self.hidden_states[int(x[1][i])] = output[i]
            self.cell_states[int(x[1][i])] = c_n[0][i]
        return self.last_layer(output.squeeze())

    def training_step(self, batch, batch_idx):
        x = [batch['features'].squeeze(), batch['investment_id'].squeeze()]
        y = batch['target'].squeeze()
        loss = F.mse_loss(self.forward(x).squeeze(), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
    # do something with all training_step outputs
        self.hidden_states = {}
        self.cell_states = {}

    def validation_epoch_end(self, val_step_outputs):
        self.hidden_states = {}
        self.cell_states = {}
    
    def validation_step(self, batch, batch_idx):
        x = [batch['features'].squeeze(), batch['investment_id'].squeeze()]
        y = batch['target'].squeeze()
        loss = F.mse_loss(torch.squeeze(self.forward(x)), y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _get_states(self, investment_ids):
        """
        Input:
            investment_ids: torch.Tensor[Batch_size, 1] - investment_ids

        Output:
            hidden_states: torch.Tensor[Batch_size, hidden_size] - hidden_states for objects in batch computed as hidden_state = self.hidden_states['investment_id']
            cell_states: torch.Tensor[Batch_size, hidden_size] - cell_states for objects in batch computed as cell_states = self.cell_states['investment_id']
        """
        hidden_states = torch.cat([self.hidden_states[int(investment_id)].unsqueeze(0) if investment_id in self.hidden_states else self.init_hidden_state for investment_id in investment_ids], dim=0)
        cell_states = torch.cat([self.cell_states[int(investment_id)].unsqueeze(0) if investment_id in self.cell_states else self.init_cell_state for investment_id in investment_ids], dim=0)
        return hidden_states.float(), cell_states.float()