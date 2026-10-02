"""NVARC synthetic-puzzle corpus converter tests."""

from __future__ import annotations

import json

from arc.serialize.prompt import parse_completion
from arc.synth.nvarc import load_nvarc_pairs, nvarc_examples, nvarc_file_to_examples


def _pair(inp, out):
    return {"input": inp, "output": out}


def _write(tmp_path, pairs):
    path = tmp_path / "0934a4d8_0607ce86.json"
    path.write_text(json.dumps(pairs))
    return path


def test_load_drops_malformed_pairs(tmp_path):
    path = _write(
        tmp_path,
        [
            _pair([[1, 2], [3, 4]], [[4, 3], [2, 1]]),
            _pair([[0, 0]], [[0, 0]]),
            {"input": [[1, 2]], "output": [[1, 99]]},  # 99 out of range -> dropped
            {"input": [[1, 2]]},                        # missing output -> dropped
        ],
    )
    loaded = load_nvarc_pairs(path)
    assert len(loaded) == 2
    assert loaded[0].input == ((1, 2), (3, 4))
    assert loaded[0].output == ((4, 3), (2, 1))


def test_examples_leave_one_out_format(tmp_path):
    pairs = load_nvarc_pairs(
        _write(tmp_path, [_pair([[1]], [[2]]), _pair([[2]], [[3]]), _pair([[3]], [[4]])])
    )
    exs = nvarc_examples(pairs, max_support=2, max_queries=None, seed=0)
    assert len(exs) == 3  # each pair becomes the query once
    for ex in exs:
        assert ex.prompt.rstrip().endswith("Output:")     # open for the target
        assert parse_completion(ex.completion) is not None  # completion parses to a grid


def test_completion_matches_query_output(tmp_path):
    pairs = load_nvarc_pairs(
        _write(tmp_path, [_pair([[1, 1]], [[2, 2]]), _pair([[3, 3]], [[4, 4]])])
    )
    outs = {parse_completion(e.completion) for e in nvarc_examples(pairs, max_queries=None)}
    assert ((2, 2),) in outs and ((4, 4),) in outs


def test_support_is_capped(tmp_path):
    pairs = load_nvarc_pairs(_write(tmp_path, [_pair([[i]], [[i + 1]]) for i in range(6)]))
    exs = nvarc_examples(pairs, max_support=2, max_queries=3, seed=1)
    assert len(exs) == 3
    for ex in exs:
        # 2 support pairs (each an "Input:/Output:" block) + the query's open "Output:"
        assert ex.prompt.count("Output:") == 3


def test_needs_at_least_two_pairs(tmp_path):
    pairs = load_nvarc_pairs(_write(tmp_path, [_pair([[1]], [[2]])]))
    assert nvarc_examples(pairs) == []


def test_file_to_examples_end_to_end(tmp_path):
    path = _write(tmp_path, [_pair([[1]], [[2]]), _pair([[3]], [[4]]), _pair([[5]], [[6]])])
    assert len(nvarc_file_to_examples(path, max_support=2, max_queries=None)) == 3


def test_kaggle_train_loads_nvarc_directory(tmp_path):
    """The trainer reads a DIRECTORY of NVARC files (converting in-kernel), one
    example per file — the retrain path that needs no derived-corpus upload."""
    import kaggle_train

    d = tmp_path / "nvarc_full" / "grp"
    d.mkdir(parents=True)
    for name in ("a", "b", "c"):
        (d / f"{name}.json").write_text(
            json.dumps([_pair([[1]], [[2]]), _pair([[3]], [[4]])])
        )
    examples = kaggle_train._load_corpus_examples(str(tmp_path))
    assert len(examples) == 3  # 1 example/file x 3 files
    assert examples[0].prompt.rstrip().endswith("Output:")


def test_kaggle_train_directory_respects_max_files(tmp_path):
    import kaggle_train

    d = tmp_path / "nvarc_full"
    d.mkdir(parents=True)
    for i in range(5):
        (d / f"{i}.json").write_text(json.dumps([_pair([[1]], [[2]]), _pair([[3]], [[4]])]))
    assert len(kaggle_train._load_corpus_examples(str(tmp_path), max_files=2)) == 2


def test_kaggle_train_still_loads_jsonl(tmp_path):
    import kaggle_train
    from arc.solvers.llm.ttt_data import TrainExample
    from arc.synth import save_examples_jsonl

    path = tmp_path / "corpus.jsonl"
    save_examples_jsonl([TrainExample("Input:\n1\nOutput:", "\n2")], path)
    assert len(kaggle_train._load_corpus_examples(str(path))) == 1
