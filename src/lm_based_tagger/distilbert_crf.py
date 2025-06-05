# distilbert_crf.py
import torch
from torchcrf import CRF
import torch.nn as nn
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