# distilbert_crf.py
import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class DistilBertCRFForTokenClassification(nn.Module):
    # Tag bigrams that never occur in the training data. They are ruled out at decode time:
    # a noun directly followed by another noun means the first one should have been NM.
    FORBIDDEN_TRANSITIONS = (("N", "N"), ("N", "NPL"), ("NPL", "NPL"))

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
    def __init__(self, num_labels: int, id2label: dict, label2id: dict, pretrained_name: str = "distilbert-base-uncased",  dropout_prob: float = 0.1, config: DistilBertConfig | None = None):
        super().__init__()

        if config is not None:
            # Loading a fine-tuned checkpoint: every weight comes from its state dict, so build
            # the encoder from the saved config instead of downloading pretrained weights first.
            self.config = config
            self.bert = DistilBertModel(config)
        else:
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

    def _constrained_crf(self):
        """
        Copy of the CRF with FORBIDDEN_TRANSITIONS set to a large negative score, used only
        for decoding. A copy (rather than editing self.crf in place) keeps concurrent
        inference safe and leaves the learned transitions untouched.
        """
        label2id = {label: int(i) for i, label in self.config.id2label.items()}
        crf = CRF(self.crf.num_tags, batch_first=True).to(self.crf.transitions.device)
        crf.load_state_dict(self.crf.state_dict())
        with torch.no_grad():
            for before, after in self.FORBIDDEN_TRANSITIONS:
                if before in label2id and after in label2id:
                    crf.transitions[label2id[before], label2id[after]] = -1e4
        return crf

    @staticmethod
    def _compact_word_positions(emissions, word_mask, tags=None):
        """
        Gather the word positions of each row into a contiguous, left-aligned sequence.

        torchcrf assumes the mask is a contiguous prefix. Word labels are interleaved with
        feature tokens, @pos_N tokens and continuation subwords, so the CRF must run over
        the compacted word sequence for its transitions to connect adjacent words.

        Example (W = word position, . = ignored):
            word_mask  [., ., W, ., W, W, .]  ->  order [2, 4, 5, ...], mask [T, T, T]

        Returns:
            emissions [B, W, C], mask [B, W], tags [B, W] or None, order [B, W], lengths [B]
        """
        lengths = word_mask.sum(dim=1)
        max_len = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
        # Stable sort puts word positions first while keeping their original order.
        order = torch.argsort((~word_mask).to(torch.int8), dim=1, stable=True)[:, :max_len]
        num_labels = emissions.size(-1)
        compact_emissions = emissions.gather(1, order.unsqueeze(-1).expand(-1, -1, num_labels))
        compact_mask = torch.arange(max_len, device=emissions.device)[None, :] < lengths[:, None]
        # torchcrf requires the first timestep to be on; only matters for rows with no words.
        compact_mask[:, 0] = True

        compact_tags = None
        if tags is not None:
            compact_tags = tags.gather(1, order)
            compact_tags = compact_tags.masked_fill(~compact_mask | (compact_tags < 0), 0)
        return compact_emissions, compact_mask, compact_tags, order, lengths

    def forward(self, input_ids=None, attention_mask=None, labels=None, word_mask=None, **kwargs):
        """
        Forward pass for training or inference.

        Args:
            input_ids (Tensor): Token IDs of shape [B, T]
            attention_mask (Tensor): Attention mask of shape [B, T]
            labels (Tensor, optional): Ground-truth labels of shape [B, T]. Required during training.
            word_mask (Tensor, optional): Bool mask of shape [B, T], True at the first subword of each
                identifier word. Required for CRF decoding at inference; derived from labels in training.
            kwargs: Any additional DistilBERT-compatible inputs (e.g., head_mask, position_ids, etc.)

        Returns:
            If labels are provided (training mode):
                dict with:
                    - loss (Tensor): scalar negative log-likelihood from CRF
                    - logits (Tensor): emission scores of shape [B, T, C]

            If labels are not provided (inference mode):
                dict with:
                    - logits (Tensor): emission scores of shape [B, T, C]
                    - predictions (List[List[int]]): one list per sequence, aligned to the
                      inner tokens (excluding [CLS] and [SEP]). Word positions hold the
                      Viterbi-decoded label IDs; other positions hold the emission argmax.

        Notes:
            - logits: [B, T, C], where B = batch size, T = sequence length, C = number of label classes
            - The CRF only sees word positions (see `_compact_word_positions`), so its transition
              scores model word-to-word tag sequences such as NM -> N or V -> NM.
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
            # 2) Word positions are exactly the positions with a real label
            #    (feature tokens, @pos_N tokens, continuation subwords, CLS/SEP and padding are -100)
            emissions, crf_mask, tags, _, _ = self._compact_word_positions(
                emission_scores.float(), labels != -100, labels
            )

            # 3) Compute CRF negative log-likelihood over the contiguous word sequence
            loss = -self.crf(emissions, tags, mask=crf_mask, reduction="mean")
            return {"loss": loss, "logits": emission_scores}

        else:
            # INFERENCE MODE
            inner_emissions = emission_scores[:, 1:-1, :]           # [B, T-2, C]
            inner_lengths = attention_mask[:, 1:-1].sum(dim=1).tolist()
            predictions = inner_emissions.argmax(dim=-1)            # fallback for non-word positions

            if word_mask is not None:
                # 2) Run Viterbi over the contiguous word sequence, then scatter back to token positions
                emissions, crf_mask, _, order, lengths = self._compact_word_positions(
                    inner_emissions.float(), word_mask[:, 1:-1].bool()
                )
                best_paths = self._constrained_crf().decode(emissions, mask=crf_mask)
                for row, path in enumerate(best_paths):
                    n_words = int(lengths[row].item())
                    if n_words:
                        predictions[row, order[row, :n_words]] = torch.tensor(
                            path[:n_words], device=predictions.device
                        )

            best_paths = [row[:n] for row, n in zip(predictions.tolist(), inner_lengths)]
            return {"logits": emission_scores, "predictions": best_paths}

    @classmethod
    def from_pretrained(cls, ckpt_dir, local=False, revision=None, **kw):
        from safetensors.torch import load_file as load_safe_file
        from huggingface_hub import hf_hub_download
        import os
        cfg = DistilBertConfig.from_pretrained(ckpt_dir, local_files_only=local, revision=revision)

        model = cls(
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            config=cfg,
            **kw,
        )

        # Preserve custom config metadata saved with the fine-tuned checkpoint.
        # This is required so inference uses the same feature layout as training.
        if hasattr(cfg, "selected_features"):
            model.config.selected_features = cfg.selected_features
            model.bert.config.selected_features = cfg.selected_features
        if hasattr(cfg, "position0_label_priors"):
            model.config.position0_label_priors = cfg.position0_label_priors
            model.bert.config.position0_label_priors = cfg.position0_label_priors
        if hasattr(cfg, "pattern_postprocessing_default"):
            model.config.pattern_postprocessing_default = cfg.pattern_postprocessing_default
            model.bert.config.pattern_postprocessing_default = cfg.pattern_postprocessing_default

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
                    revision=revision,
                    local_files_only=local
                )

            state_dict = load_safe_file(weight_path, device="cpu")

            # Resize embeddings if vocab changed (e.g. special tokens added during training)
            emb_key = "bert.embeddings.word_embeddings.weight"
            if emb_key in state_dict:
                saved_vocab = state_dict[emb_key].shape[0]
                if saved_vocab != model.bert.embeddings.word_embeddings.num_embeddings:
                    model.bert.resize_token_embeddings(saved_vocab)

            model.load_state_dict(state_dict)
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model.safetensors from {ckpt_dir}: {e}")