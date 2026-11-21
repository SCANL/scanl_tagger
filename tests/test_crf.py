import torch

from conftest import LABEL2ID, save_checkpoint
from scalar_tagger.lm_based_tagger.distilbert_crf import DistilBertCRFForTokenClassification


def test_compact_word_positions():
    emissions = torch.arange(2 * 7 * 3, dtype=torch.float).view(2, 7, 3)
    word_mask = torch.tensor([
        [0, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ], dtype=torch.bool)
    tags = torch.tensor([
        [-100, -100, 2, -100, 0, 1, -100],
        [-100, 1, -100, -100, -100, -100, -100],
    ])

    compact, mask, compact_tags, order, lengths = DistilBertCRFForTokenClassification._compact_word_positions(
        emissions, word_mask, tags
    )
    assert lengths.tolist() == [3, 1]
    assert order[0].tolist() == [2, 4, 5]
    assert order[1, 0].item() == 1
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert compact_tags[0].tolist() == [2, 0, 1]
    assert compact_tags[1].tolist() == [1, 0, 0]  # padding positions get a dummy label
    assert torch.equal(compact[0], emissions[0, [2, 4, 5]])


def _batch(tokenizer, words_per_row):
    """Tokenize rows of [@pos, word] pairs and return encoded inputs, labels and word masks."""
    encoded = tokenizer(words_per_row, is_split_into_words=True, padding=True, return_tensors="pt")
    labels = torch.full_like(encoded["input_ids"], -100)
    for row in range(len(words_per_row)):
        previous = None
        for idx, word_idx in enumerate(encoded.word_ids(batch_index=row)):
            if word_idx is not None and word_idx % 2 == 1 and word_idx != previous:
                labels[row, idx] = LABEL2ID["N"] if word_idx == len(words_per_row[row]) - 1 else LABEL2ID["NM"]
            previous = word_idx
    return encoded, labels


def test_training_loss_reaches_crf_transitions(tiny_model, tokenizer):
    tiny_model.train()
    encoded, labels = _batch(tokenizer, [["@pos_0", "max", "@pos_2", "size"], ["@pos_2", "employee"]])
    out = tiny_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], labels=labels)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert tiny_model.crf.transitions.grad is not None
    assert tiny_model.crf.transitions.grad.abs().sum() > 0


def test_inference_decodes_over_word_positions_only(tiny_model, tokenizer):
    encoded, labels = _batch(tokenizer, [["@pos_0", "max", "@pos_1", "employee", "@pos_2", "size"], ["@pos_2", "name"]])
    word_mask = labels != -100
    with torch.no_grad():
        out = tiny_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], word_mask=word_mask)

    crf = tiny_model._constrained_crf()
    for row in range(2):
        # Covers at least every real inner token (padded rows also keep the [SEP] slot)
        inner_len = int(encoded["attention_mask"][row].sum()) - 2
        assert len(out["predictions"][row]) >= inner_len

        # Viterbi over just this row's word emissions must match the word positions' predictions
        positions = word_mask[row].nonzero().flatten()
        word_emissions = out["logits"][row, positions].unsqueeze(0).float()
        expected = crf.decode(word_emissions)[0]
        assert [out["predictions"][row][p - 1] for p in positions.tolist()] == expected


def test_constrained_crf_forbids_transitions_without_touching_learned_ones(tiny_model):
    before = tiny_model.crf.transitions.detach().clone()
    constrained = tiny_model._constrained_crf()
    for a, b in DistilBertCRFForTokenClassification.FORBIDDEN_TRANSITIONS:
        assert constrained.transitions[LABEL2ID[a], LABEL2ID[b]].item() == -1e4
    assert constrained.transitions[LABEL2ID["NM"], LABEL2ID["N"]].item() == before[LABEL2ID["NM"], LABEL2ID["N"]].item()
    assert torch.equal(tiny_model.crf.transitions, before)


def test_forbidden_transition_changes_decode(tiny_model):
    # Emissions prefer N for both words; without the constraint Viterbi would return N N.
    emissions = torch.full((1, 2, len(LABEL2ID)), -5.0)
    emissions[0, :, LABEL2ID["N"]] = 5.0
    emissions[0, :, LABEL2ID["NM"]] = 4.0
    with torch.no_grad():
        tiny_model.crf.transitions.zero_()
        tiny_model.crf.start_transitions.zero_()
        tiny_model.crf.end_transitions.zero_()
    assert tiny_model.crf.decode(emissions)[0] == [LABEL2ID["N"], LABEL2ID["N"]]
    assert tiny_model._constrained_crf().decode(emissions)[0] in (
        [LABEL2ID["NM"], LABEL2ID["N"]], [LABEL2ID["N"], LABEL2ID["NM"]],
    )


def test_from_pretrained_round_trip(tiny_model, tokenizer, tmp_path):
    tiny_model.config.selected_features = ["context", "digit"]
    path = save_checkpoint(tiny_model, tokenizer, tmp_path / "ckpt")

    loaded = DistilBertCRFForTokenClassification.from_pretrained(path, local=True)
    assert loaded.config.selected_features == ["context", "digit"]
    for (name, a), (_, b) in zip(tiny_model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), name


def _brute_force_marginals(crf, emissions, n_words):
    """P(tag_t = y) by enumerating every tag sequence of one row (feasible for tiny inputs)."""
    import itertools

    num_tags = emissions.size(-1)
    scores, paths = [], list(itertools.product(range(num_tags), repeat=n_words))
    for path in paths:
        score = crf.start_transitions[path[0]] + crf.end_transitions[path[-1]]
        score = score + sum(emissions[t, path[t]] for t in range(n_words))
        score = score + sum(crf.transitions[path[t - 1], path[t]] for t in range(1, n_words))
        scores.append(score)
    probs = torch.softmax(torch.stack(scores), dim=0)
    marginals = torch.zeros(n_words, num_tags)
    for p, path in zip(probs, paths):
        for t, tag in enumerate(path):
            marginals[t, tag] += p
    return marginals


def test_marginals_match_brute_force_enumeration(tiny_model):
    torch.manual_seed(1)
    crf = tiny_model._constrained_crf()
    emissions = torch.randn(3, 3, len(LABEL2ID))
    lengths = [3, 1, 2]  # variable lengths in one batch, as in real batches
    mask = torch.arange(3)[None, :] < torch.tensor(lengths)[:, None]
    with torch.no_grad():
        marginals = DistilBertCRFForTokenClassification._marginals(crf, emissions, mask)
        for row, n in enumerate(lengths):
            expected = _brute_force_marginals(crf, emissions[row], n)
            assert torch.allclose(marginals[row, :n], expected, atol=1e-5), row
            assert torch.allclose(marginals[row, :n].sum(-1), torch.ones(n), atol=1e-5)


def test_marginals_respect_forbidden_transitions(tiny_model):
    crf = tiny_model._constrained_crf()
    emissions = torch.zeros(1, 2, len(LABEL2ID))
    emissions[0, 0, LABEL2ID["N"]] = 20.0  # first word is almost surely N
    with torch.no_grad():
        marginals = DistilBertCRFForTokenClassification._marginals(crf, emissions, torch.ones(1, 2, dtype=torch.bool))
    # N N and N NPL are forbidden, so the second word can't be N or NPL
    assert marginals[0, 1, LABEL2ID["N"]] < 1e-6
    assert marginals[0, 1, LABEL2ID["NPL"]] < 1e-6


def test_inference_returns_marginals_for_decoded_tags(tiny_model, tokenizer):
    encoded, labels = _batch(tokenizer, [["@pos_0", "max", "@pos_1", "employee", "@pos_2", "size"], ["@pos_2", "name"]])
    word_mask = labels != -100
    with torch.no_grad():
        out = tiny_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                         word_mask=word_mask, return_marginals=True)
    assert [len(row) for row in out["marginals"]] == [3, 1]
    for row in out["marginals"]:
        for word in row:
            assert abs(sum(word) - 1) < 1e-5


def test_temperature_softens_marginals_without_changing_the_best_tags(tiny_model):
    torch.manual_seed(2)
    crf = tiny_model._constrained_crf()
    emissions = torch.randn(1, 3, len(LABEL2ID)) * 5
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        raw = DistilBertCRFForTokenClassification._marginals(crf, emissions, mask)
        soft = DistilBertCRFForTokenClassification._marginals(crf, emissions, mask, temperature=2.0)
        expected = _brute_force_marginals(crf, emissions[0], 3)
    assert torch.allclose(raw[0], expected, atol=1e-5)
    assert soft.max() < raw.max()
    assert torch.allclose(soft.sum(-1), torch.ones(1, 3), atol=1e-5)
