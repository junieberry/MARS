import json
from pathlib import Path

import torch
import wandb
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LongformerModel
from wonderwords import RandomWord

from mars.collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from mars.dataloader import RecformerTrainDataset, RecformerEvalDataset
from mars import RecformerModel, MARSForSeqRec, MARSTokenizer, MARSConfig
from utils.args import parse_finetune_args
from utils.optimization import create_optimizer_and_scheduler
from utils.utils import AverageMeterSet, Ranker, load_data

wandb_logger: wandb.sdk.wandb_run.Run | None = None
tokenizer_glb: MARSTokenizer | None = None


def load_config_tokenizer(args, item2id):
    config = MARSConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.session_reduce_method = args.session_reduce_method
    config.pooler_type = args.pooler_type
    config.global_attention_type = args.global_attention_type
    config.linear_dim = args.linear_dim
    config.attribute_agg_method = args.attribute_agg_method

    tokenizer = MARSTokenizer.from_pretrained(args.model_name_or_path, config)

    if args.global_attention_type not in ["cls", "attribute"]:
        raise ValueError("Unknown global attention type.")

    return config, tokenizer


def _par_tokenize_doc(doc):
    item_id, item_attr = doc

    input_ids, token_type_ids, attr_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids, attr_type_ids


def encode_all_items(model: RecformerModel, tokenizer: MARSTokenizer, tokenized_items, args):
    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(
                range(0, len(items), args.batch_size * args.encode_item_batch_size_multiplier),
                ncols=100,
                desc="Encode all items",
        ):

            item_batch = [[item] for item in items[i: i + args.batch_size * args.encode_item_batch_size_multiplier]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            if args.pooler_type != "token":
                item_embeddings.append(outputs.pooler_output.detach())
            else:
                pooler_output = outputs.pooler_output.detach()  # (bs, 1, max_seq_len, hidden_size)
                pooler_output = pooler_output.permute(0, 2, 1, 3)  # (bs, max_seq_len, 1, hidden_size)
                for j in range(pooler_output.shape[0]):
                    output_ = pooler_output[j]  # (max_seq_len, 1, hidden_size)
                    item_embeddings.append(output_)

    if args.pooler_type == "token":
        item_embeddings = torch.nn.utils.rnn.pad_sequence(
            item_embeddings, batch_first=True, padding_value=float("nan")
        )  # (bs, max_seq_len, 1, hidden_size)
    else:
        item_embeddings = torch.cat(item_embeddings, dim=0)  # (bs, attr_num, 1, hidden_size)

    return item_embeddings


def evaluate(model, dataloader, args, return_preds=False):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    all_scores = []
    all_labels = []

    for batch, labels in tqdm(dataloader, ncols=100, desc="Evaluate"):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad(), autocast(dtype=torch.bfloat16, enabled=args.bf16):
            scores = model(**batch)  # (bs, |I|, num_attr, items_max)

        all_scores.append(scores.detach().clone().cpu())
        all_labels.append(labels.detach().clone().cpu())

        assert torch.isnan(scores).sum() == 0, "NaN in scores."

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2 * i]
            metrics["Recall@%d" % k] = res[2 * i + 1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    if return_preds:
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0).squeeze()
        all_predictions = torch.topk(all_scores, k=max(args.metric_ks), dim=1).indices
        return average_metrics, all_predictions, all_labels

    return average_metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, args, train_step: int):
    global wandb_logger

    epoch_losses = []

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc="Training")):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with autocast(dtype=torch.bfloat16, enabled=args.bf16):
            loss = model(**batch)

        if torch.any(torch.isnan(loss)):
            continue

        if wandb_logger is not None:
            wandb_logger.log({f"train_step_{train_step}/loss": loss.item()})
            epoch_losses.append(loss.item())

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update learning rate schedule

    if wandb_logger is not None:
        wandb_logger.log({f"train_step_{train_step}/epoch_loss": sum(epoch_losses) / len(epoch_losses)})


def main(args):
    print(args)

    seed_everything(args.seed, workers=True)

    train, val, test, item_meta_dict, item2id, id2item, user2id, id2user = load_data(args)
    config, tokenizer = load_config_tokenizer(args, item2id)
    global tokenizer_glb
    tokenizer_glb = tokenizer

    if args.random_word is None:
        random_word_generator = RandomWord()
        while True:
            random_word = random_word_generator.random_words(include_parts_of_speech=["noun", "verb"])[0]

            if " " in random_word or "-" in random_word:
                continue
            else:
                break
    else:
        random_word = args.random_word

    path_corpus = Path(args.data_path)
    path_output = Path(args.output_dir) / random_word

    try:
        path_output.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        raise FileExistsError(f"Output directory ({path_output}) already exists.")

    global wandb_logger
    wandb_logger = wandb.init(
        project="MARS",
        group=args.wandb_group_name or path_corpus.name,
        config=vars(args),
    )

    doc_tuples = [
        _par_tokenize_doc(doc) for doc in tqdm(item_meta_dict.items(), ncols=100, desc=f"[Tokenize] {path_corpus}")
    ]
    tokenized_items = {
        item2id[item_id]: [input_ids, token_type_ids, attr_type_ids]
        for item_id, input_ids, token_type_ids, attr_type_ids in doc_tuples
    }

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode="val", collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode="test", collator=eval_data_collator)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(
        val_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=val_data.collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * args.eval_test_batch_size_multiplier, collate_fn=test_data.collate_fn
    )

    longformer_model = LongformerModel.from_pretrained(args.model_name_or_path)
    model = MARSForSeqRec(config)
    model.longformer.embeddings.load_state_dict(longformer_model.embeddings.state_dict())
    model.longformer.encoder.load_state_dict(longformer_model.encoder.state_dict())
    del longformer_model
    model.to(args.device)

    if args.fix_word_embedding:
        print("Fix word embeddings.")
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)

    model.init_item_embedding(item_embeddings)
    model.to(args.device)  # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    test_metrics = evaluate(model, test_loader, args)
    if wandb_logger is not None:
        wandb_logger.log({f"zero-shot/{k}": v for k, v in test_metrics.items()})
    print(f"Test set Zero-shot: {test_metrics}")

    if args.zero_shot_only:
        return

    best_target = float("-inf")
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, args, 1)

        if epoch + 1:
            dev_metrics = evaluate(model, dev_loader, args)
            print(f"Epoch: {epoch}. Dev set: {dev_metrics}")

            if wandb_logger is not None:
                wandb_logger.log({f"dev_step_1/{k}": v for k, v in dev_metrics.items()})

            if dev_metrics["NDCG@10"] > best_target:
                print("Save the best model.")
                best_target = dev_metrics["NDCG@10"]
                patient = 5
                torch.save(model.state_dict(), path_output / "stage_1_best.pt")

            else:
                patient -= 1
                if patient == 0:
                    break

    print("Load best model in stage 1.")
    model.load_state_dict(torch.load(path_output / "stage_1_best.pt"))

    test_metrics = evaluate(model, test_loader, args)
    print(f"Stage-1 Test set: {test_metrics}")
    if wandb_logger is not None:
        wandb_logger.log({f"stage_1_test/{k}": v for k, v in test_metrics.items()})

    if not args.one_step_training:
        patient = 3

        for epoch in range(args.num_train_epochs):

            train_one_epoch(model, train_loader, optimizer, scheduler, args, 2)

            if epoch + 1:
                dev_metrics = evaluate(model, dev_loader, args)
                print(f"Epoch: {epoch}. Dev set: {dev_metrics}")

                if wandb_logger is not None:
                    wandb_logger.log({f"dev_step_2/{k}": v for k, v in dev_metrics.items()})

                if dev_metrics["NDCG@10"] > best_target:
                    print("Save the best model.")
                    best_target = dev_metrics["NDCG@10"]
                    patient = 3
                    torch.save(model.state_dict(), path_output / "stage_2_best.pt")

                else:
                    patient -= 1
                    if patient == 0:
                        break

        print("Load best model in stage 2.")
        try:
            model.load_state_dict(torch.load(path_output / "stage_2_best.pt"))
        except FileNotFoundError:
            print("No best model in stage 2. Use the latest model.")

        test_metrics, predictions, labels = evaluate(model, test_loader, args, return_preds=True)
        print(f"Stage-2 Test set: {test_metrics}")

        if wandb_logger is not None:
            wandb_logger.log({f"stage_2_test/{k}": v for k, v in test_metrics.items()})

        users = list(map(int, test.keys()))
        users = list(map(id2user.get, users))

        predictions = predictions.tolist()
        labels = labels.tolist()

        output = {}
        for user, prediction, label in zip(users, predictions, labels):
            prediction = list(map(id2item.get, prediction))
            label = id2item[label]
            output[user] = {"predictions": prediction, "target": label}

        json.dump(output, open(path_output / "predictions.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(parse_finetune_args())
