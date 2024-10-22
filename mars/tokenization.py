from collections import defaultdict

import torch
from transformers import LongformerTokenizer


class IntFactory:
    def __init__(self):
        self.counter = 0

    def __call__(self) -> int:
        self.counter += 1
        return self.counter


class MARSTokenizer(LongformerTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None):
        cls.config = config
        cls.attr_map: defaultdict[str, int] = defaultdict(IntFactory())  # Mapping attribute name to id
        return super().from_pretrained(pretrained_model_name_or_path)

    def __call__(self, items, pad_to_max=False, return_tensor=False):
        """
        items: item sequence or a batch of item sequence, item sequence is a list of dict
        pad_to_max: whether to pad to max length
        return_tensor: whether to return tensor

        return:
        input_ids: token ids
        item_position_ids: the position of items
        token_type_ids: id for key or value
        attention_mask: local attention masks
        global_attention_mask: global attention masks for Longformer
        """

        if len(items) > 0 and isinstance(items[0], list):  # batched items
            inputs = self.batch_encode(items, pad_to_max=pad_to_max)

        else:
            inputs = self.encode(items)

        if return_tensor:

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v)

        return inputs

    def item_tokenize(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def encode_item(self, item):

        input_ids = []
        token_type_ids = []
        attr_type = []  # 1 for title, 2 for brand, etc.  0 is ignored
        item = list(item.items())[: self.config.max_attr_num]  # truncate attribute number

        for attribute in item:
            attr_name, attr_value = attribute

            name_tokens = self.item_tokenize(attr_name)
            value_tokens = self.item_tokenize(attr_value)

            attr_tokens = name_tokens + value_tokens
            attr_tokens = attr_tokens[: self.config.max_attr_length]

            input_ids += attr_tokens

            attr_type_ids = [1] * len(name_tokens)
            attr_type_ids += [2] * len(value_tokens)
            attr_type_ids = attr_type_ids[: self.config.max_attr_length]
            attr_type_ = [self.attr_map[attr_name]] * len(attr_tokens)
            attr_type_ = attr_type_[: self.config.max_attr_length]

            token_type_ids += attr_type_ids
            attr_type += attr_type_

        return input_ids, token_type_ids, attr_type

    def encode(
        self,
        items,
        encode_item=True,
    ):
        """
        Encode a sequence of items.
        the order of items:  [past...present]
        return: [present...past]
        """
        items = items[::-1]  # reverse items order
        items = items[: self.config.max_item_embeddings - 1]  # truncate the number of items, -1 for <s>

        input_ids = [self.bos_token_id]
        item_position_ids = [0]
        token_type_ids = [0]
        attr_type_ids = [0]

        for item_idx, item in enumerate(items):

            if encode_item:

                item_input_ids, item_token_type_ids, item_attr_type_ids = self.encode_item(item)

            else:
                item_input_ids, item_token_type_ids, item_attr_type_ids = item

            input_ids += item_input_ids
            token_type_ids += item_token_type_ids
            attr_type_ids += item_attr_type_ids

            item_position_ids += [item_idx + 1] * len(item_input_ids)  # item_idx + 1 make idx starts from 1 (0 for <s>)

        input_ids = input_ids[: self.config.max_token_num - 1]
        item_position_ids = item_position_ids[: self.config.max_token_num - 1]
        token_type_ids = token_type_ids[: self.config.max_token_num - 1]
        attr_type_ids = attr_type_ids[: self.config.max_token_num - 1]

        input_ids += [self.eos_token_id]
        item_position_ids += [0]  # len(items) + 1 for <s>
        token_type_ids += [0]
        attr_type_ids += [0]

        attention_mask = [1] * len(input_ids)
        if self.config.global_attention_type == "cls":
            global_attention_mask = [0] * len(input_ids)
            global_attention_mask[0] = 1
        elif self.config.global_attention_type == "attribute":
            global_attention_mask = [1 if a != 2 else 0 for a in token_type_ids]  # 0 for bos, 1 for type, 2 for value
            assert len(global_attention_mask) == len(input_ids)
        else:
            raise ValueError("Unknown global attention type.")

        return {
            "input_ids": input_ids,
            "item_position_ids": item_position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "attr_type_ids": attr_type_ids,
        }

    def padding(self, item_batch, pad_to_max):

        if pad_to_max:
            max_length = self.config.max_token_num
        else:
            max_length = max([len(items["input_ids"]) for items in item_batch])

        batch_input_ids = []
        batch_item_position_ids = []
        batch_attention_mask = []
        batch_global_attention_mask = []
        batch_attr_type_ids = []

        for items in item_batch:
            input_ids = items["input_ids"]
            item_position_ids = items["item_position_ids"]
            attention_mask = items["attention_mask"]
            global_attention_mask = items["global_attention_mask"]
            item_attr_type_ids = items["attr_type_ids"]

            length_to_pad = max_length - len(input_ids)

            input_ids += [self.pad_token_id] * length_to_pad
            item_position_ids += [self.config.max_item_embeddings - 1] * length_to_pad
            attention_mask += [0] * length_to_pad
            global_attention_mask += [0] * length_to_pad
            item_attr_type_ids += [0] * length_to_pad

            batch_input_ids.append(input_ids)
            batch_item_position_ids.append(item_position_ids)
            batch_attention_mask.append(attention_mask)
            batch_global_attention_mask.append(global_attention_mask)
            batch_attr_type_ids.append(item_attr_type_ids)

        return {
            "input_ids": batch_input_ids,
            "item_position_ids": batch_item_position_ids,
            "attention_mask": batch_attention_mask,
            "global_attention_mask": batch_global_attention_mask,
            "attr_type_ids": batch_attr_type_ids,
        }

    def batch_encode(self, item_batch, encode_item=True, pad_to_max=False):

        item_batch = [self.encode(items, encode_item) for items in item_batch]

        return self.padding(item_batch, pad_to_max)
