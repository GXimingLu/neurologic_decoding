import json
import math
import argparse
import torch
import logging
import numpy as np
import pickle
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel

from mt.generate import generate
from mt import utils_seq2seq
from lexical_constraints import init_batch

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")

    parser.add_argument("--input_file", type=str, help="source file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument("--output_file", type=str, help="output file")

    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=20,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=5,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.1,
                        help="length penalty for beam search")

    parser.add_argument('--prune_factor', type=float, default=1.5,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    args = parser.parse_args()

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    eos_ids = [model.config.eos_token_id]
    for period in ['.', '▁.']:
        eos_ids.append(tokenizer.encoder.get(period))

    input_lines = [l.strip() for l in open(args.input_file, 'r').readlines()]
    items = pickle.load(open(args.constraint_file, 'rb'))
    constraints_list = utils_seq2seq.load_constraint(tokenizer, items)

    output_lines = []
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            src_text = input_lines[next_i:next_i + args.batch_size]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            next_i += args.batch_size

            inputs = tokenizer.prepare_translation_batch(src_text)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
            outputs = generate(self=model,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               min_length=args.min_tgt_length,
                               max_length=args.max_tgt_length,
                               no_repeat_ngram_size=args.ngram_size,
                               num_beams=args.beam_size,
                               length_penalty=args.length_penalty,
                               constraints=constraints,
                               prune_factor=args.prune_factor,
                               sat_tolerance=args.sat_tolerance)

            output_sequences = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            output_lines.extend(output_sequences)
            pbar.update(1)

    original_lines = [l.strip() for l in open('../data/input.txt', 'r').readlines()]
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i, o in zip(original_lines, output_lines):
            fout.write(f'{i} ||| {o}')
            fout.write("\n")


if __name__ == "__main__":
    main()
