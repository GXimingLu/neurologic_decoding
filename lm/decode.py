import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from lm.generate import generate
from unilm import utils_seq2seq
from lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")

    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")

    parser.add_argument('--prune_factor', type=float, default=50,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    args = parser.parse_args()

    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id

    with open(args.input_path) as fin:
        input_lines = [line.split('=')[0] + "=" for line in fin.read().splitlines()]

    with open(args.constraint_file, 'r') as f:
        constraints_list = []
        for line in f:
            constraints = []
            for concept in json.loads(line):
                constraints.append([f' {c}' for c in concept])
            constraints_list.append(constraints)

    input_lines = [tokenizer.tokenize(x) for x in input_lines]
    constraints_list = utils_seq2seq.tokenize_constraints(tokenizer, constraints_list)
    input_lines = sorted(list(enumerate(input_lines)),
                         key=lambda x: len(x[1]))
    index = [x[0] for x in input_lines]
    constraints_list = [constraints_list[i] for i in index]
    output_lines = [""] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            full_chunk = input_lines[next_i:next_i + args.batch_size]
            prompt_tokens_num = sorted(set([len(x[1]) for x in full_chunk]))
            step_len = args.batch_size
            if len(prompt_tokens_num) > 1:
                step_len = len([x for x in full_chunk if len(x[1]) == prompt_tokens_num[0]])

            _chunk = input_lines[next_i:next_i + step_len]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + step_len],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            next_i += step_len

            input_ids = torch.stack([torch.from_numpy(np.array(tokenizer.convert_tokens_to_ids(x))) for x in buf])
            input_ids = input_ids.to('cuda')

            outputs = generate(self=model,
                               input_ids=input_ids,
                               min_length=args.min_tgt_length,
                               max_length=args.max_tgt_length,
                               num_beams=args.beam_size,
                               no_repeat_ngram_size=args.ngram_size,
                               length_penalty=args.length_penalty,
                               constraints=constraints,
                               prune_factor=args.prune_factor,
                               sat_tolerance=args.sat_tolerance)

            prompt = [tokenizer.convert_tokens_to_string(x) for x in buf]
            output_sequences = [tokenizer.decode(o).split('<|endoftext|>')[0].split(prompt[i])[-1].replace('=', '').strip()
                                for i, o in enumerate(outputs)]

            for i in range(len(buf)):
                output_lines[buf_id[i]] = output_sequences[i]
            pbar.update(1)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")


if __name__ == "__main__":
    main()
