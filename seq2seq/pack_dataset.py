"""Fill examples with bitext up to max_tokens without breaking up examples.
[['I went', 'yo fui'],
['to the store', 'a la tienda']
]
=> ['I went to the store', 'yo fui a la tienda']
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from transformers import AutoTokenizer


def pack_examples(tok, src_examples, tgt_examples, max_tokens=1024):

    finished_src, finished_tgt = [], []

    sorted_examples = list(sorted(zip(src_examples, tgt_examples), key=lambda x: len(x[0])))
    new_src, new_tgt = sorted_examples[0]

    def is_too_big(strang):
        return tok(strang, return_tensors="pt").input_ids.shape[1] > max_tokens

    for src, tgt in tqdm(sorted_examples[1:]):
        cand_src = new_src + " " + src
        cand_tgt = new_tgt + " " + tgt
        if is_too_big(cand_src) or is_too_big(cand_tgt):  # cant fit, finalize example
            finished_src.append(new_src)
            finished_tgt.append(new_tgt)
            new_src, new_tgt = src, tgt
        else:  # can fit, keep adding
            new_src, new_tgt = cand_src, cand_tgt
        # import ipdb; ipdb.set_trace()

    # cleanup
    if new_src:
        assert new_tgt
        finished_src.append(new_src)
        finished_tgt.append(new_tgt)
    return finished_src, finished_tgt


def minify(src_dir: Path, dest_dir: Path, n: int):
    """Write first n lines of each file f in src_dir to dest_dir/f"""
    dest_dir.mkdir(exist_ok=True)
    for path in src_dir.iterdir():
        new = [x.rstrip() for x in list(path.open().readlines())][:n]
        dest_path = dest_dir.joinpath(path.name)
        print(dest_path)
        dest_path.open("w").write("\n".join(new))


def pack_data_dir(tok, data_dir: Path, max_tokens, save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    for split in ["val", "test", "train"]:
        src_path, tgt_path = data_dir / f"{split}.source", data_dir / f"{split}.target"
        src_docs = [x.rstrip() for x in Path(src_path).open().readlines()]
        tgt_docs = [x.rstrip() for x in Path(tgt_path).open().readlines()]
        packed_src, packed_tgt = pack_examples(tok, src_docs, tgt_docs, max_tokens)
        print(f"packed {split} split from {len(src_docs)} examples -> {len(packed_src)}.")
        Path(save_path / f"{split}.source").open("w").write("\n".join(packed_src))
        Path(save_path / f"{split}.target").open("w").write("\n".join(packed_tgt))


def packer_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tok_name)
    return pack_data_dir(tokenizer, Path(args.data_dir), args.max_seq_len, args.save_path)


if __name__ == "__main__":
    packer_cli()