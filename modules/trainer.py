import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, label_ranking_loss


class Trainer(object):
    def __init__(self, model, train_dataloader, valid_dataloader, config, device):
        self.model = model
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.config = config
        self.device = device

    def build_model(self):
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        t_total = len(self.train_loader) * self.config["epoch"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["warmup_step"],
            num_training_steps=t_total,
        )

    def train(self):
        best_val_loss = 1e10
        scaler = GradScaler()
        self.model.zero_grad()
        is_save = False
        print(
            f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val F1':^9} | {'is_save':^9} "
        )
        for epoch in range(self.config["epoch"]):
            self.model.train()
            t_loss = 0
            for step, batch in enumerate(self.train_loader):
                cv_batch = batch[0].to(self.device)
                nlp_inputs = batch[1].to(self.device)
                nlp_attentions = batch[2].to(self.device)
                labels = batch[3].to(self.device)
                batch_dict = {"input_ids": nlp_inputs, "attention_mask": nlp_attentions}
                with autocast():
                    outputs = self.model(batch_dict, cv_batch)
                loss = self.loss(outputs, labels)
                t_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                # exploding gradients를 방지
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                self.scheduler.step()
                scaler.update()
                self.model.zero_grad()

            val_loss, val_f1 = self.eval()
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                is_save = True
                torch.save(self.model.state_dict(), "./model/model_res.pt")
            print(
                f"{epoch + 1:^7} | {t_loss / len(self.train_loader):^12} | {val_loss:^10} | {val_f1:^9} | {is_save:^9} "
            )
            is_save = False
        return

    def eval(self):
        self.model.eval()
        pred_list = []
        label_list = []
        fin_loss = 0

        for batch in self.valid_loader:
            cv_batch = batch[0].to(self.device)
            nlp_inputs = batch[1].to(self.device)
            nlp_attentions = batch[2].to(self.device)
            batch_dict = {"input_ids": nlp_inputs, "attention_mask": nlp_attentions}
            labels = batch[3].to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_dict, cv_batch)
            loss = self.loss(outputs, labels)
            fin_loss += loss.item()
            pred = torch.argmax(outputs, dim=1).flatten().detach().cpu().numpy().tolist()
            labels = labels.flatten().detach().cpu().numpy().tolist()
            pred_list += pred
            label_list += labels

        f1_scores = f1_score(label_list, pred_list, average="weighted")

        return fin_loss / len(self.valid_loader), f1_scores
