from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader

from functions.dataloader import HDF5Dataset, collate_pad
from transformer_model.img_cnn_encoder import ImageEncoder
from transformer_model.transformer import Transformer
from functions.nlg_metrics import Metrics
from functions.train_utils import seed_everything, load_json
from functions.test_utils import parse_arguments
from functions.gpu_cuda_helper import select_device

if __name__ == "__main__":
    args = parse_arguments()

    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    config = load_json(args.config_path)

    dataset_dir = args.dataset_dir
    dataset_dir = Path(dataset_dir)
    images_path = str(dataset_dir / "test_images.hdf5")
    captions_path = str(dataset_dir / "test_captions.json")
    lengths_path = str(dataset_dir / "test_lengths.json")
    checkpoints_dir = config["paths"]["checkpoint"]
    checkpoint_name = args.checkpoint_name

    SEED = config["seed"]
    seed_everything(SEED)

    print("loading dataset...")
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    sos_id = vocab.stoi["<sos>"]
    eos_id = vocab.stoi["<eos>"]
    vocab_size = len(vocab)

    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    test_dataset = HDF5Dataset(hdf5_path=images_path,
                               captions_path=captions_path,
                               lengthes_path=lengths_path,
                               pad_id=pad_id,
                               transform=transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    max_len = config["max_len"]
    val_iter = DataLoader(test_dataset,
                          collate_fn=collate_pad(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=4,
                          shuffle=False)
    print("loading dataset finished.")

    print("constructing transformer_model.\n")
    # prepare some hyper parameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    h, w = image_enc_hyperparms["encode_size"], image_enc_hyperparms[
        "encode_size"]
    image_seq_len = int(image_enc_hyperparms["encode_size"] ** 2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    image_enc = ImageEncoder(**image_enc_hyperparms)
    transformer = Transformer(**transformer_hyperparms)

    load_path = str(Path(checkpoints_dir) / checkpoint_name)
    state = torch.load(load_path, map_location=torch.device("cpu"))
    image_model_state = state["transformer_model"][0]
    transformer_state = state["transformer_model"][1]
    image_enc.load_state_dict(image_model_state)
    transformer.load_state_dict(transformer_state)

    image_enc.to(device).eval()
    transformer.to(device).eval()

    eval_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "log_prob": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    selected_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    nlgMetrics = Metrics()
    bleu4 = []
    pb = tqdm(val_iter, leave=False, total=len(val_iter))
    pb.unit = "step"
    for imgs, cptns_all, lens in pb:
        imgs: Tensor
        cptns_all: Tensor
        lens: Tensor

        k = 5
        imgs = imgs.to(device)
        start = torch.full(size=(1, 1),
                           fill_value=sos_id,
                           dtype=torch.long,
                           device=device)
        with torch.no_grad():
            imgs_enc = image_enc(imgs)
            logits, attns = transformer(imgs_enc, start)
            logits: Tensor
            attns: Tensor
            log_prob = F.log_softmax(logits, dim=2)
            log_prob_topk, indexes_topk = log_prob.topk(k, sorted=True)
            current_preds = torch.cat(
                [start.expand(k, 1), indexes_topk.view(k, 1)], dim=1)

        seq_preds = []
        seq_log_probs = []
        seq_attns = []
        while current_preds.size(1) <= (
                max_len - 2) and k > 0 and current_preds.nelement():
            with torch.no_grad():
                imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                logits, attns = transformer(imgs_expand, current_preds)
                log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                log_prob = log_prob + log_prob_topk.view(k, 1)
                log_prob_topk, indexes_topk = log_prob.view(-1).topk(k, sorted=True)
                prev_seq_k, next_word_id = np.unravel_index(indexes_topk.cpu(), log_prob.size())
                next_word_id = torch.as_tensor(next_word_id).to(device).view(k, 1)
                current_preds = torch.cat((current_preds[prev_seq_k], next_word_id), dim=1)

            seqs_end = (next_word_id == eos_id).view(-1)
            if torch.any(seqs_end):
                seq_preds.extend(seq.tolist() for seq in current_preds[seqs_end])
                seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())
                k -= torch.sum(seqs_end)
                current_preds = current_preds[~seqs_end]
                log_prob_topk = log_prob_topk[~seqs_end]

        specials = [pad_id, sos_id, eos_id]
        seq_preds, seq_attns, seq_log_probs = zip(
            *sorted(zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))

        text_preds = [[vocab.itos[s] for s in seq if s not in specials] for seq in seq_preds]
        text_refs = [[vocab.itos[r] for r in ref if r not in specials] for ref in cptns_all.squeeze(0).permute(1, 0)]

        scores = defaultdict(list)
        for text_pred in text_preds:
            for k, v in nlgMetrics.calculate([text_refs], [text_pred]).items():
                scores[k].append(v)

        eval_data["hypos_text"].append(text_preds)
        eval_data["refs_text"].append(text_refs)
        eval_data["attns"].append(list(seq_attns))
        eval_data["log_prob"].append(list(seq_log_probs))
        for k, v_list in scores.items():
            eval_data[k].append(v_list)

        selected_data["hypos_text"].append(text_preds[0])
        selected_data["refs_text"].append(text_refs)
        selected_data["attns"].append(list(seq_attns)[0])
        selected_data["bleu1"].append(scores["bleu1"][0])
        selected_data["bleu2"].append(scores["bleu2"][0])
        selected_data["bleu3"].append(scores["bleu3"][0])
        selected_data["bleu4"].append(scores["bleu4"][0])
        selected_data["gleu"].append(scores["gleu"][0])
        selected_data["meteor"].append(scores["meteor"][0])

        pb.set_description(
            f'bleu4: Current: {selected_data["bleu4"][-1]:.4f}, Max: {max(selected_data["bleu4"]):.4f}, Min: {min(selected_data["bleu4"]):.4f}, Mean: {mean(selected_data["bleu4"]):.4f} \u00B1 {pstdev(selected_data["bleu4"]):.2f}'
        )

    pb.close()

    print("\nSaving data...")
    experiment_name = checkpoint_name.split("/")[0]
    save_dir = Path(args.save_dir) / f"{experiment_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=eval_data).to_json(str(save_dir / "all.json"))
    pd.DataFrame(data=selected_data).to_json(
        str(save_dir / "selected.json"))

    print("The output file is saved.")
