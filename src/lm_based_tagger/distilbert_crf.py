# distilbert_crf.py
import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class DistilBertCRFForTokenClassification(nn.Module):
    """
    Token-level classifier that combines DistilBERT with a CRF layer for structured prediction.

    Architecture:
        input_ids, attention_mask
            ↓
        DistilBERT (pretrained encoder)
            ↓
        Dropout
            ↓
        Linear layer (projects hidden size → num_labels)
            ↓
        CRF layer (models sequence-level transitions)

    Training:
        - Uses negative log-likelihood from CRF as loss.
        - Learns both emission scores (token-level confidence) and
          transition scores (label-to-label sequence consistency).

    Inference:
        - Uses Viterbi decoding to predict the most likely sequence of labels.

    Output:
        During training:
            {"loss": ..., "logits": ...}
        During inference:
            {"logits": ..., "predictions": List[List[int]]}

    Example input shape:
        input_ids:      [B, T]      — e.g. [16, 128]
        attention_mask: [B, T]      — 1 for real tokens, 0 for padding
        logits:         [B, T, C]   — C = number of label classes
    """
    def __init__(self, num_labels: int, id2label: dict, label2id: dict, pretrained_name: str = "distilbert-base-uncased",  dropout_prob: float = 0.1):
        super().__init__()

        self.config = DistilBertConfig.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.bert = DistilBertModel.from_pretrained(pretrained_name, config=self.config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for training or inference.

        Args:
            input_ids (Tensor): Token IDs of shape [B, T]
            attention_mask (Tensor): Attention mask of shape [B, T]
            labels (Tensor, optional): Ground-truth labels of shape [B, T]. Required during training.
            kwargs: Any additional DistilBERT-compatible inputs (e.g., head_mask, position_ids, etc.)

        Returns:
            If labels are provided (training mode):
                dict with:
                    - loss (Tensor): scalar negative log-likelihood from CRF
                    - logits (Tensor): emission scores of shape [B, T, C]

            If labels are not provided (inference mode):
                dict with:
                    - logits (Tensor): emission scores of shape [B, T, C]
                    - predictions (List[List[int]]): decoded label IDs from CRF,
                                                    one list per sequence,
                                                    each of length T-2 (excluding [CLS] and [SEP])

        Notes:
            - logits: [B, T, C], where B = batch size, T = sequence length, C = number of label classes
            - predictions: List[List[int]], where each inner list has length T-2
                        (i.e., excludes [CLS] and [SEP]) and contains Viterbi-decoded label IDs
        """

        # Hugging Face occasionally injects helper fields (e.g. num_items_in_batch)
        # Filter `kwargs` down to what DistilBertModel.forward actually accepts.
        ALLOWED = {
            "head_mask", "inputs_embeds", "position_ids",
            "output_attentions", "output_hidden_states", "return_dict"
        }
        bert_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED}

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs,
        )
        # 1) Compute per-token emission scores
        # Applies dropout to the BERT hidden states, then projects them to label logits.
        # Shape: [B, T, C], where B=batch size, T=sequence length, C=number of classes
        sequence_output = self.dropout(outputs[0])
        emission_scores = self.classifier(sequence_output)

        if labels is not None:
            # 2) Remove [CLS] and [SEP] special tokens from emissions and labels
            # These tokens were added by the tokenizer but are not part of the identifier
            emissions = emission_scores[:, 1:-1, :]         # [B, T-2, C]
            tags      = labels[:, 1:-1].clone()             # [B, T-2]

            # 3) Create a mask: True where label is valid, False where label == -100
            # The CRF will use this to ignore special/padded tokens
            crf_mask  = (tags != -100)

            # 4) Replace invalid label positions (-100) with a dummy label (e.g., 0)
            # This is required because CRF expects a label at every position, even if masked
            tags[~crf_mask] = 0

            # 5) Ensure the first token of every sequence is active in the CRF mask
            # This avoids CRF errors when the first token is masked out (which breaks decoding)
            first_off = (~crf_mask[:, 0]).nonzero(as_tuple=True)[0]
            if len(first_off):
                crf_mask[first_off, 0] = True
                tags[first_off, 0] = 0  # assign a dummy label

            # 6) Compute CRF negative log-likelihood loss
            loss = -self.crf(emissions, tags, mask=crf_mask, reduction="mean")
            return {"loss": loss, "logits": emission_scores}

        else:
            # INFERENCE MODE

            # 2) Remove [CLS] and [SEP] from emissions and build CRF mask from attention
            # Only use the inner content of the input sequence
            crf_mask  = attention_mask[:, 1:-1].bool()      # [B, T-2]
            emissions = emission_scores[:, 1:-1, :]         # [B, T-2, C]

            # 3) Run Viterbi decoding to get best label sequence for each input
            best_paths = self.crf.decode(emissions, mask=crf_mask)
            return {"logits": emission_scores, "predictions": best_paths}
    
    @classmethod
    def from_pretrained(cls, ckpt_dir, local=False, **kw):
        from safetensors.torch import load_file as load_safe_file
        from huggingface_hub import hf_hub_download
        import os
        cfg = DistilBertConfig.from_pretrained(ckpt_dir, local_files_only=local)

        model = cls(
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            pretrained_name=cfg._name_or_path or "distilbert-base-uncased",
            **kw,
        )

        # Attempt to load model.safetensors only
        try:
            if os.path.isdir(ckpt_dir):
                # Load from local directory
                weight_path = os.path.join(ckpt_dir, "model.safetensors")
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(f"No model.safetensors found in local path: {weight_path}")
            else:
                # Load from Hugging Face Hub
                weight_path = hf_hub_download(
                    repo_id=ckpt_dir,
                    filename="model.safetensors",
                    local_files_only=local
                )

            state_dict = load_safe_file(weight_path, device="cpu")
            model.load_state_dict(state_dict)
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model.safetensors from {ckpt_dir}: {e}")