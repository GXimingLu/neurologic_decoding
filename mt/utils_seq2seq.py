

def tokenize_constraints(tokenizer, raw_cts, type):
    def tokenize(phrase):
        phrase = phrase.replace('l’', "l'").replace('L’', "L'")
        token_ids = [tokenizer.encoder.get(x) for x in tokenizer.spm_target.EncodeAsPieces(phrase)]
        assert all([x is not None for x in token_ids]), f'unrecognized token in {phrase} {type}'
        return token_ids, type
    return [[list(map(tokenize, clause)) for clause in ct] for ct in raw_cts]


def load_constraint(tokenizer, items):
    positive = tokenize_constraints(tokenizer, items['positive'], True)
    negative = tokenize_constraints(tokenizer, items['negative'], False)
    constraints = [positive[i] + negative[i] for i in range(len(positive))]
    return constraints
