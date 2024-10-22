from argparse import ArgumentParser
from pathlib import Path


def parse_finetune_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear_dim", type=int, default=256)
    parser.add_argument("--wandb_group_name", type=str, default=None)

    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="val.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--user2id_file", type=str, default="umap.json")
    parser.add_argument("--item2id_file", type=str, default="smap.json")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Model
    parser.add_argument(
        "--global_attention_type",
        type=str,
        default="cls",
        choices=["cls", "attribute"],
        help=(
            "Global attention type. 'cls' for applying global attention on CLS token, "
            "'attribute' for applying global attention on attribute name tokens."
        ),
    )
    parser.add_argument(
        "--pooler_type",
        type=str,
        default="attribute",
        choices=["attribute", "item", "token", "cls"],
        help=(
            "Pooling type. "
            "'attribute' for pooling hidden states attribute-wise, "
            "'item' for pooling hidden states item-wise, "
            "'token' for not applying any pooling, "
            "'cls' for pooling hidden states on CLS token."
        ),
    )
    parser.add_argument(
        "--session_reduce_method",
        type=str,
        default="maxsim",
        choices=["maxsim", "mean"],
        help="How to reduce session hidden states into a single vector. "
        "'maxsim' for selecting the item within the sequence that has the highest score relative to the target item, "
        "'mean' for averaging all scores relative to the target item",
    )

    parser.add_argument(
        "--attribute_agg_method",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to aggregate scores of each attribute. 'mean' for averaging, 'max' for taking the max.",
    )

    # Train
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature for softmax.")
    parser.add_argument("--num_train_epochs", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 20, 50], help="ks for Metric@k")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fix_word_embedding", action="store_true")
    parser.add_argument("--one_step_training", action="store_true")
    parser.add_argument("--eval_test_batch_size_multiplier", type=int, default=1)
    parser.add_argument("--encode_item_batch_size_multiplier", type=int, default=2)
    parser.add_argument("--random_word", type=str, default=None)
    parser.add_argument("--zero_shot_only", action="store_true")
    return parser.parse_args()
