import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")

    parser.add_argument("--input_file", type=str, help="source file")
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

    args = parser.parse_args()

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    input_lines = [l.strip() for l in open(args.input_file, 'r').readlines()]
    output_lines = []
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            src_text = input_lines[next_i:next_i + args.batch_size]
            next_i += args.batch_size

            inputs = tokenizer.prepare_translation_batch(src_text)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     min_length=args.min_tgt_length,
                                     max_length=args.max_tgt_length,
                                     no_repeat_ngram_size=args.ngram_size,
                                     num_beams=args.beam_size,
                                     length_penalty=args.length_penalty
                                     )
            output_sequences = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            output_lines.extend(output_sequences)
            pbar.update(1)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i, o in zip(input_lines, output_lines):
            fout.write(f'{i} ||| {o}')
            fout.write("\n")


if __name__ == "__main__":
    main()