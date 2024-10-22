import random
from dataclasses import dataclass
from typing import Union, List, Dict

import torch

from mars import MARSTokenizer


@dataclass
class FinetuneDataCollatorWithPadding:

    tokenizer: MARSTokenizer
    tokenized_items: Dict

    def __call__(
        self, batch_item_ids: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        attr_type_ids: (batch_size, seq_len)
        """

        batch_item_seq, labels = self.sample_train_data(batch_item_ids)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)
        batch["labels"] = labels

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)

        return batch

    def sample_train_data(self, batch_item_ids):
        batch_item_seq = []
        labels = []

        for item_ids in batch_item_ids:
            item_ids = item_ids["items"]
            item_seq_len = len(item_ids)

            assert len(item_ids) >= 2

            start_item_pos = 0
            target_item_pos = random.randrange(start_item_pos + 1, item_seq_len, 1)

            batch_item_seq.append(item_ids[:target_item_pos])
            labels.append(item_ids[target_item_pos])

        return batch_item_seq, labels

    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids, attr_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids, attr_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):

        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features


@dataclass
class EvalDataCollatorWithPadding:

    tokenizer: MARSTokenizer
    tokenized_items: Dict

    def __call__(
        self, batch_data: List[Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        """

        batch_item_seq, labels = self.prepare_eval_data(batch_data)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)

        labels = torch.LongTensor(labels)

        return batch, labels

    def prepare_eval_data(self, batch_data):

        batch_item_seq = []
        labels = []

        for data_line in batch_data:

            item_ids = data_line["items"]
            label = data_line["label"]

            batch_item_seq.append(item_ids)
            labels.append(label)

        return batch_item_seq, labels

    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids, attr_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids, attr_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):

        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features
