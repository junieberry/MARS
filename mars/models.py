from dataclasses import dataclass
from typing import List, Union, Optional, Tuple, Literal

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerModel
from transformers.models.longformer.modeling_longformer import (
    LongformerConfig,
    LongformerPreTrainedModel,
    LongformerEncoder,
    LongformerBaseModelOutputWithPooling,
    LongformerEmbeddings,
)


class MARSConfig(LongformerConfig):
    def __init__(
            self,
            attention_window: Union[List[int], int] = 64,
            sep_token_id: int = 2,
            token_type_size: int = 4,  # <s>, key, value, <pad>
            max_token_num: int = 2048,
            max_item_embeddings: int = 32,  # 1 for <s>, 50 for items
            max_attr_num: int = 12,
            max_attr_length: int = 8,
            pooler_type: str = "cls",
            temp: float = 0.05,
            mlm_weight: float = 0.1,
            item_num: int = 0,
            **kwargs,
    ):
        super().__init__(attention_window, sep_token_id, **kwargs)

        self.token_type_size = token_type_size
        self.max_token_num = max_token_num
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.pooler_type = pooler_type
        self.temp = temp
        self.mlm_weight = mlm_weight

        # finetune config
        self.item_num = item_num


@dataclass
class MARSBaseModelOutputWithPooling(LongformerBaseModelOutputWithPooling):
    mask: Optional[torch.Tensor] = None


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class MARSPooler(nn.Module):
    def __init__(self, config: MARSConfig):
        super().__init__()
        assert config.pooler_type in ["cls", "token", "item", "attribute"]

        self.pooler_type = config.pooler_type
        self.pad_token_id = config.pad_token_id

        self.linear = nn.Linear(config.hidden_size, config.linear_dim)

    def forward(
            self,
            attention_mask: torch.Tensor,
            hidden_states: torch.Tensor,
            attr_type_ids: torch.Tensor,
            item_position_ids: torch.Tensor,
    ):
        hidden_states = self.linear.forward(hidden_states)

        if self.pooler_type == "attribute":
            attr_max = attr_type_ids.max()
            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values
            items_max = num_items.max()

            attr_mask = torch.eq(
                attr_type_ids.unsqueeze(1), torch.arange(1, attr_max + 1, device=attr_type_ids.device).reshape(1, -1, 1)
            )  # (bs, attr_num, seq_len)
            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )
            attr_item_mask = torch.mul(attr_mask.unsqueeze(2), item_mask.unsqueeze(1))  # Ignore tokens that are False

            hidden_states_pooled = hidden_states.unsqueeze(1).unsqueeze(2) * attr_item_mask.unsqueeze(
                -1
            )  # (bs, attr_num, items_max, seq_len, hidden_size)

            # Sum across the required dimension
            summed_states = torch.sum(hidden_states_pooled, dim=3)  # Sum across the sequence length dimension

            # Count the number of valid (not masked out) elements in the attr_item_mask for each position
            valid_counts = attr_item_mask.sum(dim=3)
            valid_counts.unsqueeze_(-1)  # Adding an extra dimension to match the dimensionality for division

            valid_counts_eq_0 = torch.eq(valid_counts, 0)  # (bs, attr_num, items_max, 1)

            # Avoid division by zero by replacing 0 counts with 1
            valid_counts = torch.where(valid_counts_eq_0, torch.ones_like(valid_counts), valid_counts)

            # Compute the mean
            hidden_states_pooled = summed_states / valid_counts  # (bs, attr_num, items_max, hidden_size)

            return hidden_states_pooled, valid_counts_eq_0.squeeze(-1)

        elif self.pooler_type == "item":
            num_items = item_position_ids.clone()
            num_items[num_items == 50] = -100
            num_items = torch.max(num_items, dim=1).values  # (bs, )
            items_max = num_items.max()

            item_mask = torch.eq(
                item_position_ids.unsqueeze(1),
                torch.arange(1, items_max + 1, device=item_position_ids.device).reshape(1, -1, 1),
            )  # (bs, item_num, seq_len)

            hidden_states_pooled = hidden_states.unsqueeze(1) * item_mask.unsqueeze(
                -1
            )  # (bs, item_num, seq_len, hidden_size)

            # Sum across the required dimension
            summed_states = torch.sum(hidden_states_pooled, dim=2)  # Sum across the sequence length dimension

            # Instead of using NaNs, use 0s and then use the mask to compute the mean
            valid_counts = item_mask.sum(dim=2).unsqueeze(
                -1
            )  # Count of True values for the mean along seq_len dimension

            valid_counts_eq_0 = torch.eq(valid_counts, 0)  # (bs, item_num, 1)

            # Avoid division by zero by replacing 0 counts with 1
            valid_counts = torch.where(valid_counts_eq_0, torch.ones_like(valid_counts), valid_counts)

            # Compute the mean
            hidden_states_pooled = summed_states / valid_counts  # (bs, item_num, hidden_size)

            return hidden_states_pooled.unsqueeze(1), valid_counts_eq_0.squeeze(-1).unsqueeze(1)

        elif self.pooler_type == "token":
            seq_len = hidden_states.shape[1]

            mask = attention_mask[..., :seq_len].bool()  # (bs, seq_len)
            hidden_states_pooled = hidden_states  # (bs, seq_len, hidden_size)
            hidden_states_pooled[~mask] = torch.nan  # (bs, seq_len, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1)  # (bs, 1, seq_len, hidden_size)

            return hidden_states_pooled, mask

        elif self.pooler_type == "cls":
            hidden_states_pooled = hidden_states[:, 0, :]  # (bs, hidden_size)
            hidden_states_pooled = hidden_states_pooled.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, hidden_size)

            return hidden_states_pooled, torch.zeros(
                hidden_states_pooled.shape[:-1], dtype=torch.bool, device=hidden_states_pooled.device
            )

        else:
            raise ValueError(f"pooler_type {self.pooler_type} is not supported")


class RecformerModel(LongformerPreTrainedModel):
    def __init__(self, config: MARSConfig):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = MARSPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor,
            item_position_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if item_position_ids is None:
                unpadded_item_position_ids = None
            else:
                unpadded_item_position_ids = item_position_ids
                item_position_ids = nn.functional.pad(item_position_ids, (0, padding_len), value=pad_token_id)

            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0
        else:
            unpadded_item_position_ids = item_position_ids

        return (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            item_position_ids,
            inputs_embeds,
            unpadded_item_position_ids,
        )

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            attr_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            item_position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MARSBaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            item_position_ids,
            inputs_embeds,
            unpadded_item_position_ids,
        ) = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
                                                :, 0, 0, :
                                                ]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output, mask = (
            self.pooler.forward(
                attention_mask=attention_mask,
                hidden_states=sequence_output,
                attr_type_ids=attr_type_ids,
                item_position_ids=unpadded_item_position_ids,
            )
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return MARSBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
            mask=mask,
        )


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config: MARSConfig):
        super().__init__()
        self.temp = config.temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MARSForSeqRec(LongformerPreTrainedModel):
    def __init__(self, config: MARSConfig):
        super().__init__(config)

        self.longformer = RecformerModel(config)
        self.sim = Similarity(config)
        self.item_embedding = None

        self.post_init()

    def init_item_embedding(self, embeddings: torch.Tensor):
        self.item_embedding = nn.Parameter(embeddings, requires_grad=False)

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is not None:
            raise NotImplementedError("Negative sampling disabled")

        candidate_embeddings = self.item_embedding  # (|I|, attr_num, 1, hidden_size)
        pooler_output = pooler_output.unsqueeze(1)  # (batch_size, 1, attr_num, items_max, hidden_size)
        sim = self.sim(pooler_output, candidate_embeddings)  # (batch_size, |I|, attr_num, items_max)

        return sim

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            attr_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            item_position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,  # target item ids
    ):
        outputs = self.longformer.forward(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            attr_type_ids=attr_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output  # (bs, attr_num, items_max, hidden_size)
        pooler_output_mask = outputs.mask  # (bs, attr_num, items_max)  True for valid tokens

        scores = self.similarity_score(pooler_output)  # (bs, |I|, attr_num, items_max)

        all_item_mask = torch.zeros(
            (scores.shape[1], scores.shape[2], 1), dtype=torch.bool, device=scores.device
        )  # (|I|, attr_num, 1)
        final_mask = torch.add(pooler_output_mask.unsqueeze(1), all_item_mask.unsqueeze(0))
        scores[final_mask] = -torch.inf
        scores = self.reduce_session(scores)

        if labels is None:
            return scores

        loss_fct = CrossEntropyLoss()

        if labels.dim() == 2:
            labels = labels.squeeze(dim=-1)
        loss = loss_fct(scores, labels)

        return loss

    def reduce_session(self, scores: torch.Tensor):
        """
        Mask tensor: True to mask
        """
        scores: torch.Tensor  # (bs, |I|, attr_num, items_max)

        session_reduce_method: Literal["maxsim", "mean"]
        attribute_agg_method: Literal["mean", "max"]

        if self.config.session_reduce_method == "maxsim":
            # Replace NaN with -inf
            scores = scores.max(dim=-1).values  # (bs, |I|, num_attr)
        elif self.config.session_reduce_method == "mean":
            scores = scores.nanmean(dim=-1)  # (bs, |I|, num_attr)
        else:
            raise ValueError("Unknown session reduce method.")

        if self.config.attribute_agg_method == "mean":
            scores = scores.mean(dim=-1)
        elif self.config.attribute_agg_method == "max":
            scores = scores.max(dim=-1).values
        else:
            raise ValueError("Unknown attribute aggregation method.")

        return scores
