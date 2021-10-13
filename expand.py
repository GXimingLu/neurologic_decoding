import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, required=True,
                        help='dev or test')
    parser.add_argument("--input_file", type=str, required=True,
                        help='raw generated file to expand')
    parser.add_argument("--output_file", type=str, required=True,
                        help='name of expanded file')

    args = parser.parse_args()

    raw_file = open(args.input_file, 'r')
    factor_file = open(f'dataset/clean/commongen.{args.split}.factor.txt', 'r')
    output_file = open(args.output_file, 'a')

    for l, f in tqdm(zip(raw_file, factor_file)):
        for i in range(int(f)):
            output_file.write(l)


