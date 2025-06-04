# distilbert_crf.py
import torch, os
import torch.nn as nn
from TorchCRF import CRF
from transformers import DistilBertModel, DistilBertConfig

class DistilBertCRFForTokenClassification(nn.Module):
    """
    DistilBERT ➜ dropout ➜ linear projection ➜ CRF.
    The CRF layer models label‑to‑label transitions, so the model
    is optimised at *sequence* level rather than *token* level.
    """
    def __init__(self,
                 num_labels: int,
                 id2label: dict,
                 label2id: dict,
                 pretrained_name: str = "distilbert-base-uncased",
                 dropout_prob: float = 0.1):
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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):

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
        # —— Build emissions once ——————————————————————————————
        sequence_output = self.dropout(outputs[0])          # [B, T, H]
        emission_scores = self.classifier(sequence_output)  # [B, T, C]

        seq_len = emission_scores.size(1)                   # original token length

        # ============================== TRAINING ==============================
        if labels is not None:
            # 1. Drop [CLS] (idx 0) and [SEP] (idx –1)
            emissions = emission_scores[:, 1:-1, :]         # [B, T‑2, C]
            tags      = labels[:,           1:-1].clone()   # [B, T‑2]
            crf_mask  = (tags != -100)                      # True = keep

            # 2. For any position that’s masked‑off ➜ set tag to a valid id (0)
            tags[~crf_mask] = 0

            # 3. Guarantee first timestep is ON for every sequence
            first_off = (~crf_mask[:, 0]).nonzero(as_tuple=True)[0]
            if len(first_off):
                crf_mask[first_off, 0] = True        # flip mask to ON
                tags[first_off, 0] = 0               # give it tag 0

            loss = -self.crf(emissions, tags, mask=crf_mask, reduction="mean")
            return {"loss": loss, "logits": emission_scores}

        # ============================= INFERENCE ==============================
        else:
            crf_mask  = attention_mask[:, 1:-1].bool()      # [B, T‑2]
            emissions = emission_scores[:, 1:-1, :]         # [B, T‑2, C]
            best_paths = self.crf.decode(emissions, mask=crf_mask)
            return {"logits": emission_scores,
                    "predictions": best_paths}

    from transformers import DistilBertConfig
    @classmethod
    def from_pretrained(cls, ckpt_dir, **kw):
        from safetensors import safe_open
        cfg = DistilBertConfig.from_pretrained(ckpt_dir)
        model = cls(
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            pretrained_name=cfg._name_or_path or "distilbert-base-uncased",
            **kw,
        )

        weight_path_pt  = os.path.join(ckpt_dir, "pytorch_model.bin")
        weight_path_safe = os.path.join(ckpt_dir, "model.safetensors")

        if os.path.exists(weight_path_pt):
            state = torch.load(weight_path_pt, map_location="cpu")
        elif os.path.exists(weight_path_safe):
            state = {}
            with safe_open(weight_path_safe, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
        else:
            raise FileNotFoundError("No weight file found in checkpoint directory.")

        model.load_state_dict(state)
        return model