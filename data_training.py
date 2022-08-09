from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from transformer_model.img_cnn_encoder import ImageEncoder
from transformer_model.transformer import Transformer

from functions.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from functions.train_utils import parse_arguments, load_json
from functions.gpu_cuda_helper import select_device
from datetime import datetime
from typing import Dict, List, Union, Optional
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from functions.nlg_metrics import Metrics
from functions.custom_types import ModelType, OptimType, DeviceTye, DataIterType
from functions.custom_types import SchedulerType
from functions.train_utils import seed_everything


def load_datasets(dataset_processed_dir: str, pid_pad: float):
    # Setting some paths
    dataset_processed_dir = Path(dataset_processed_dir)
    images_train_path = dataset_processed_dir / "train_images.hdf5"
    images_val_path = dataset_processed_dir / "val_images.hdf5"
    captions_train_path = dataset_processed_dir / "train_captions.json"
    captions_val_path = dataset_processed_dir / "val_captions.json"
    lengths_train_path = dataset_processed_dir / "train_lengths.json"
    lengths_val_path = dataset_processed_dir / "val_lengths.json"

    # images transform
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])
    # load datasets with HDF5
    train_dataset = HDF5Dataset(hdf5_path=images_train_path,
                                captions_path=captions_train_path,
                                lengthes_path=lengths_train_path,
                                pad_id=pid_pad,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengths_val_path,
                              pad_id=pid_pad,
                              transform=transform)

    return train_dataset, val_dataset

class TrackMetrics:

    def __init__(self) -> None:

        self.reset_running()
        self.metrics = self.init_metrics()

    def create_default_dict(self):

        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    def reset_running(self):
        self.running = self.create_default_dict()

    def init_metrics(self):
        return self.create_default_dict()

    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def update(self, phase: str):
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()


class Trainer():

    def __init__(self,
                 optims: List[OptimType],
                 schedulers: List[SchedulerType],
                 device: DeviceTye,
                 epochs: int,
                 val_interval: int,
                 early_stop: int,
                 lr_patience: int,
                 embeddings_finetune: int,
                 grad_clip: float,
                 lambda_c: float,
                 checkpoints_path: str,
                 pad_id: int,
                 resume: Optional[str] = None) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop training if the model doesn't improve for n-validation epochs
        self.stop = early_stop
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        self.lr_patience = lr_patience
        # start tune embeddings after n training epochs have beed passed
        self.finetune_embedding = embeddings_finetune
        self.pad_id = pad_id

        # criterion, optimizer and schedulers
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]
        self.image_scheduler = schedulers[0]
        self.transformer_scheduler = schedulers[1]

        # metrics functions and tracker
        self.nlgMetrics = Metrics()
        self.metrics_tracker = TrackMetrics()
        self.best_metric = 0

        # Some coefficient of Doubly stochastic attention regularization
        self.lc = lambda_c
        self.grad_clip_c = grad_clip  # gradient clip coefficient

        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent
            print(time_tag)
        # Tensorboard writer
        self.logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/logs")
        self.loss_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/loss")
        self.bleu4_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/bleu4")
        self.gleu_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/gleu")

        # make folder for the experiment
        checkpoints_path = Path(checkpoints_path) / f"{time_tag}"  # type: Path
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_path)

    def loss_fn(self, logits: Tensor, targets: Tensor, attns: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))

        # Doubly stochastic attention regularization:
        # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
        # change atten size to be
        # [layer_num, head_num, batch_size, max_len, encode_size^2]
        attns = attns.permute(0, 2, 1, 3, 4)
        ln, hn = attns.size()[:2]  # number of layers, number of heads

        # calc λ(1-∑αi)^2 for each pixel in each head in each layer
        # alphas [layer_num, head_num, batch_size*encode_size^2]
        # TODO:
        # Reduction: Would it make any difference if I sum across
        # (encode_size^2, and head) dimensions and average across batch and
        # layers?
        alphas = self.lc * (1. - attns.sum(dim=3).view(ln, hn, -1)) ** 2
        alphas: Tensor
        dsar = alphas.mean(-1).sum()

        return loss + dsar

    def clip_gradient(self):
        for optim in [self.img_embed_optim, self.transformer_optim]:
            for group in optim.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.grad_clip_c,
                                               self.grad_clip_c)

    def remove_pad(
            self, tensor: Tensor, lens: Tensor,
            mask: Tensor) -> Union[List[List[List[str]]], List[List[str]]]:
        # format 3D tensor (References) to 2D tensor
        lens = lens.view(-1)
        max_len = tensor.size(1)
        is3d = len(tensor.size()) == 3
        if is3d:
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
            mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

        # Remove pads: select elements that are not equal to pad (into 1d
        # tensor) then split the formed 1d tensor according to lengthes
        tensor = torch.masked_select(tensor, mask=mask)
        tensor = torch.split(tensor, split_size_or_sections=lens.tolist())
        tensor = [[str(e.item()) for e in t] for t in tensor]

        # Get back into 3d (list of list of list)
        if is3d:
            tensor = [tensor[i:i + 5] for i in range(0, len(tensor), 5)]

        return tensor

    def get_metrics(self, gtruth: Tensor, lens: Tensor, preds: Tensor):
        # gtruth [B, lm - 1, cn=5]
        # lens [B, cn=5]
        # preds [B, lm]
        mask = gtruth != self.pad_id  # mask pad tokens
        refs = self.remove_pad(gtruth, lens, mask)
        hypos = self.remove_pad(preds, lens[:, 0], mask[:, :, 0])

        scores = self.nlgMetrics.calculate(refs, hypos, self.train)

        return scores

    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)

    def check_improvement(self, metric: float):
        is_better = metric > self.best_metric
        reduce_lr = False
        es = False
        if is_better:
            self.best_metric = metric
            self.bad_epochs_num = 0
        else:
            self.bad_epochs_num += 1

        if self.bad_epochs_num > self.lr_patience:
            reduce_lr = True
            self.num_bad_epochs = 0

        if self.bad_epochs_num > self.stop:
            es = True  # early stop

        return is_better, reduce_lr, es

    def load_checkpoint(self):
        load_path = str(Path(self.checkpoints_path).parent / self.resume)

        # load checkpoint
        state = torch.load(load_path, map_location=torch.device("cpu"))
        image_model_state = state["transformer_model"][0]
        transformer_state = state["transformer_model"][1]
        image_optim_state = state["optims"][0]
        transformer_optim_state = state["optims"][1]
        image_scheduler_state = state["schedulers"][0]
        transformer_scheduler_state = state["schedulers"][1]

        # load state dicts
        self.img_embed_optim.load_state_dict(image_optim_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.image_scheduler.load_state_dict(image_scheduler_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]
        self.best_metric = state["best_metric"]
        self.metrics_tracker.running = state["running_metrics"]
        self.metrics_tracker.metrics = state["metrics"]

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        return image_model_state, transformer_state

    def save_checkpoint(self, models: List[ModelType], is_best: bool):

        image_model_state = models[0].state_dict()
        Transformer_state = models[1].state_dict()
        image_optim_state = self.img_embed_optim.state_dict()
        transformer_optim_state = self.transformer_optim.state_dict()
        image_scheduler_state = self.image_scheduler.state_dict()
        transformer_scheduler_state = self.transformer_scheduler.state_dict()

        state = {
            "transformer_model": [image_model_state, Transformer_state],
            "optims": [image_optim_state, transformer_optim_state],
            "schedulers": [image_scheduler_state, transformer_scheduler_state],
            "phase": self.train,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
            "best_metric": self.best_metric,
            "running_metrics": self.metrics_tracker.running,
            "metrics": self.metrics_tracker.metrics
        }

        # set save path
        file_name = "checkpoint"
        if is_best:
            file_name = f"{file_name}_best"
        save_path = Path(self.checkpoints_path) / f"{file_name}.pth.tar"

        torch.save(state, save_path)

    def plot_data(self, phase, metrics_dict):
        current_metric = {f"{phase}": metrics_dict["loss"][-1]}
        self.loss_logger.add_scalars("loss", current_metric, self.epoch)

        current_metric = {f"{phase}": metrics_dict["bleu4"][-1]}
        self.bleu4_logger.add_scalars("bleu4", current_metric, self.epoch)

        current_metric = {f"{phase}": metrics_dict["gleu"][-1]}
        self.gleu_logger.add_scalars("gleu", current_metric, self.epoch)

        # Plot all metrics except loss and bleu4
        if not self.train:
            ex = ["loss", "bleu4", "gleu"]
            name_tag = "Metrics Validation"
            current_metrics = {
                k: v[-1]
                for k, v in metrics_dict.items() if k not in ex
            }
            self.logger.add_scalars(name_tag, current_metrics, self.epoch)

    def run(self, img_embeder: ModelType, transformer: ModelType,
            data_iters: DataIterType, SEED: int):
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        if self.resume is not None:
            model_state_dicts = self.load_checkpoint()
            print("load the resume.")
            img_embeder.load_state_dict(model_state_dicts[0])
            transformer.load_state_dict(model_state_dicts[1])

        # move transformer_model to device
        img_embeder = img_embeder.to(self.device)
        transformer = transformer.to(self.device)

        # start
        main_pb = tqdm(range(1, self.epochs_num + 1))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {(self.epoch + 1):02d}")

            is_best = False
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                img_embeder.train()
                transformer.train()
                data_iter = data_iters[0]
                # fine tune the embeddings layer after some epochs and add the
                # parameters to the optimizer
                if self.epoch == self.finetune_embedding:
                    for p in transformer.decoder.cptn_emb.parameters():
                        p.requires_grad = True
                        self.transformer_optim.add_param_group({"params": p})
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # Iterate over data
            # progress bar
            data_iter
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengths of all captions [B, cn]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step + 1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)

                # zero the parameter gradients
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # embed images using CNN then get logits prediction using
                    # the transformer
                    imgs = img_embeder(imgs)
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # get predictions then calculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predictions
                targets = cptns_all[:, 1:]  # remove <SOS>
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # step ended
                # update progress bar
                # pb.update(1)

            self.metrics_tracker.update(phases[self.train])  # save metrics
            if not self.train:
                checked_metric = self.metrics_tracker.metrics["val"]["bleu4"]
                is_best, lr_r, es = self.check_improvement(checked_metric[-1])

                if lr_r:  # reduce lr
                    self.image_scheduler.step()
                    self.transformer_scheduler.step()

            # save checkpoint
            if self.train or is_best:
                self.save_checkpoint(models=[img_embeder, transformer],
                                     is_best=is_best)

            # PLot metrics
            phase = phases[self.train]
            metrics_dict = self.metrics_tracker.metrics[phase]
            self.plot_data(phase, metrics_dict)

            # epoch ended
            self.set_phase()  # set train or vall phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                break

if __name__ == "__main__":

    # parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # MS COCO hdf5 and json files
    resume = args.resume
    if resume == "":
        resume = None

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load configuration file
    config = load_json(args.config_path)

    # load vocab
    min_freq = config["min_freq"]
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    vocab_size = len(vocab)

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading processed dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_params"]
    max_len = config["max_len"]
    train_ds, val_ds = load_datasets(dataset_dir, pad_id)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_padd(max_len, pad_id),
                            pin_memory=True,
                            **loader_params)
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_padd(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=0,
                          shuffle=True)
    print("loading processed dataset finished.")
    print(f"number of vocabulary is {vocab_size}\n")

    # --------------- Construct transformer_model, optimizers --------------- #
    print("constructing transformer_model")
    # prepare some hyper-parameters
    image_enc_hyperparams = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparams["encode_size"] ** 2)

    transformer_hyperparams = config["hyperparams"]["transformer"]
    transformer_hyperparams["vocab_size"] = vocab_size
    transformer_hyperparams["pad_id"] = pad_id
    transformer_hyperparams["img_encode_size"] = image_seq_len
    transformer_hyperparams["max_len"] = max_len - 1

    # construct transformer_model
    image_enc = ImageEncoder(**image_enc_hyperparams)
    image_enc.fine_tune(True)
    transformer = Transformer(**transformer_hyperparams)

    # load pretrained embeddings
    print("loading pretrained glove embeddings...")
    weights = vocab.vectors
    transformer.decoder.cptn_emb.from_pretrained(weights,
                                                 freeze=True,
                                                 padding_idx=pad_id)
    list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False

    # Optimizers and schedulers
    image_enc_lr = config["optim_params"]["encoder_lr"]
    params2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = Adam(params=params2update, lr=image_enc_lr)
    gamma = config["optim_params"]["lr_factors"][0]
    image_scheduler = StepLR(image_encoder_optim, step_size=1, gamma=gamma)

    transformer_lr = config["optim_params"]["transformer_lr"]
    params2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = Adam(params=params2update, lr=transformer_lr)
    gamma = config["optim_params"]["lr_factors"][1]
    transformer_scheduler = StepLR(transformer_optim, step_size=1, gamma=gamma)

    # --------------- Training --------------- #
    print("start training...\n")
    train = Trainer(optims=[image_encoder_optim, transformer_optim],
                    schedulers=[image_scheduler, transformer_scheduler],
                    device=device,
                    pad_id=pad_id,
                    resume=resume,
                    checkpoints_path=config["paths"]["checkpoint"],
                    **config["train_params"])
    train.run(image_enc, transformer, [train_iter, val_iter], SEED)

    print("done")

