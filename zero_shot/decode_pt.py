import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from transformers import AutoTokenizer, AutoModelWithLMHead

import utils
from zero_shot.generate import generate
from lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument("--input_path", type=str, help="initialization of decoding")

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

    parser.add_argument('--lambda_1', type=float, default=10,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    args = parser.parse_args()
    print(args)

    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġwho', "'s"]
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

    with open(args.input_path) as fin:
        input_lines = [line.strip() for line in fin.read().splitlines()]

    def read_constraints(file_name):
        cons_list = []
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                cons = []
                for concept in json.loads(line):
                    cons.append([f' {c}' for c in concept if c.islower()])
                cons_list.append(cons)
        return cons_list

    constraints_list = read_constraints(args.constraint_file)

    input_lines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines]
    constraints_list = utils.tokenize_constraints(tokenizer, constraints_list)

    if path.exists(args.output_file):
        count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        input_lines = input_lines[count:]
        constraints_list = constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids,
                                     input_id_list=_chunk)
            buf = _chunk
            next_i += args.batch_size

            max_len = max([len(x) for x in buf])
            buf = [x + [PAD_ID] * (max_len - len(x)) for x in buf]

            input_ids = torch.stack([torch.from_numpy(np.array(x)) for x in buf])
            input_ids = input_ids.to('cuda')
            attention_mask = (~torch.eq(input_ids, PAD_ID)).int()
            attention_mask = attention_mask.to('cuda')

            advanced_constraints = []
            for j, init_cons in enumerate(constraints):
                adv_cons = init_cons
                for token in _chunk[j // args.beam_size]:
                    adv_cons = adv_cons.advance(token)
                advanced_constraints.append(adv_cons)

            outputs, _, _ = generate(self=model,
                                     input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     pad_token_id=PAD_ID,
                                     bad_words_ids=bad_words_ids,
                                     min_length=args.min_tgt_length,
                                     max_length=args.max_tgt_length,
                                     num_beams=args.beam_size,
                                     no_repeat_ngram_size=args.ngram_size,
                                     length_penalty=args.length_penalty,
                                     constraints=advanced_constraints,
                                     lambda_1=args.lambda_1,
                                     sat_tolerance=args.sat_tolerance)

            prompt = [tokenizer.decode(x) for x in buf]
            output_sequences = [prompt[i] + tokenizer.decode(o).split(prompt[i])[-1].split('<|endoftext|>')[0].rstrip()
                                for i, o in enumerate(outputs)]

            for hypothesis in output_sequences:
                fout.write(hypothesis.strip().replace('<|endoftext|>', '') + "\n")
                fout.flush()

            pbar.update(1)

if __name__ == "__main__":
    main()
