import numpy as np
import torch
from operator import attrgetter
from concurrent import futures
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Set, Union

from lexical_constraints import ConstrainedHypothesis, ConstrainedCandidate


def topk_huggingface_parallel(timestep: int,
                              batch_size: int,
                              beam_size: int,
                              vocab_size: int,
                              pad_token_id: int,
                              prune_factor: float,
                              sat_tolerance: float,
                              inactive: np.array,
                              scores: np.array,
                              hypotheses: List[ConstrainedHypothesis],
                              num_fill: int) -> Tuple[np.array, np.array, List[List[Union[ConstrainedHypothesis, None]]], List[List[int]]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param vocab_size: The size of vocabulary.
    :param pad_token_id:
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: (batch_size, beam_size * target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size,))
    :param num_fill: The number of required return beam
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    seq_scores, raw_token_idx = torch.topk(scores, beam_size, dim=1, largest=True, sorted=True)
    best_ids = (raw_token_idx // vocab_size).cpu().numpy()
    best_word_ids = (raw_token_idx % vocab_size).cpu().numpy()
    seq_scores = seq_scores.cpu().numpy()

    scores = torch.reshape(scores, [batch_size, beam_size, -1]).cpu().numpy()

    select_best_ids = np.ones((batch_size, num_fill)) * -1
    select_best_word_ids = np.ones((batch_size, num_fill)) * -1
    select_seq_scores = np.zeros((batch_size, num_fill))
    select_hypotheses = [[None] * num_fill for _ in range(batch_size)]
    select_num_mets = [[-1] * num_fill for _ in range(batch_size)]

    # if timestep < 50:
    input_args = []
    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        if all([x is None for x in hypotheses[rows]]):
            select_best_ids[sentno] = [0] * num_fill
            select_best_word_ids[sentno] = [pad_token_id] * num_fill
            select_seq_scores[sentno] = [0] * num_fill
            select_hypotheses[sentno] = [None] * num_fill
            select_num_mets[sentno] = [-1] * num_fill
            continue

        assert not any([x is None for x in hypotheses[rows]]), 'Bad state'

        input_args.append((sentno, timestep, beam_size, prune_factor, sat_tolerance, inactive[sentno], scores[sentno],
                           hypotheses[rows], best_ids[sentno], best_word_ids[sentno], seq_scores[sentno], num_fill))

    def fill(result):
        index = result[0]
        select_best_ids[index], select_best_word_ids[index], select_seq_scores[index], \
            select_hypotheses[index], select_num_mets[index] = result[1], result[2], result[3], result[4], result[5]

    with futures.ProcessPoolExecutor() as pool:
        for output in pool.map(sequential_topk, input_args):
            fill(output)

    select_raw_token_idx = select_best_ids * vocab_size + select_best_word_ids
    return select_seq_scores, select_raw_token_idx, select_hypotheses, select_num_mets

def topk_huggingface(timestep: int,
                     batch_size: int,
                     beam_size: int,
                     vocab_size: int,
                     pad_token_id: int,
                     prune_factor: float,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     num_fill: int) -> Tuple[np.array, np.array, List[List[Union[ConstrainedHypothesis, None]]], List[List[int]]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param vocab_size: The size of vocabulary.
    :param pad_token_id:
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: (batch_size, beam_size * target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size,))
    :param num_fill: The number of required return beam
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    seq_scores, raw_token_idx = torch.topk(scores, beam_size, dim=1, largest=True, sorted=True)
    best_ids = (raw_token_idx // vocab_size).cpu().numpy()
    best_word_ids = (raw_token_idx % vocab_size).cpu().numpy()
    seq_scores = seq_scores.cpu().numpy()

    scores = torch.reshape(scores, [batch_size, beam_size, -1]).cpu().numpy()

    select_best_ids = np.ones((batch_size, num_fill)) * -1
    select_best_word_ids = np.ones((batch_size, num_fill)) * -1
    select_seq_scores = np.zeros((batch_size, num_fill))
    select_hypotheses = [[None] * num_fill for _ in range(batch_size)]
    select_num_mets = [[-1] * num_fill for _ in range(batch_size)]

    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        if all([x is None for x in hypotheses[rows]]):
            select_best_ids[sentno] = [0] * num_fill
            select_best_word_ids[sentno] = [pad_token_id] * num_fill
            select_seq_scores[sentno] = [0] * num_fill
            select_hypotheses[sentno] = [None] * num_fill
            select_num_mets[sentno] = [-1] * num_fill
            continue

        assert not any([x is None for x in hypotheses[rows]]), 'Bad state'

        _, select_best_ids[sentno], select_best_word_ids[sentno], select_seq_scores[sentno],\
            select_hypotheses[sentno], select_num_mets[sentno] = _sequential_topk(sentno,
                                                                                  pad_token_id,
                                                                                  beam_size,
                                                                                  prune_factor,
                                                                                  sat_tolerance,
                                                                                  inactive[sentno],
                                                                                  scores[sentno],
                                                                                  hypotheses[rows],
                                                                                  best_ids[sentno],
                                                                                  best_word_ids[sentno],
                                                                                  seq_scores[sentno],
                                                                                  num_fill=num_fill)

    select_raw_token_idx = select_best_ids * vocab_size + select_best_word_ids
    return select_seq_scores, select_raw_token_idx, select_hypotheses, select_num_mets


def sequential_topk(args):
    return _sequential_topk(*args)


def _sequential_topk(sentno: int,
                     timestep: int,
                     beam_size: int,
                     prune_factor: float,
                     sat_tolerance: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: np.array,
                     best_word_ids: np.array,
                     sequence_scores: np.array,
                     num_fill: int) -> Tuple[int, np.array, np.array, np.array,
                                             List[ConstrainedHypothesis], List[int]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param timestep: The current decoder timestep.
    :param beam_size: The length of the beam for each segment.
    :param inactive: Array listing inactive rows (shape: (beam_size,)).
    :param scores: The scores array (shape: (beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (beam_size,))
    :param best_ids: The current list of best hypotheses (shape: (beam_size,)).
    :param best_word_ids: The parallel list of best word IDs (shape: (beam_size,)).
    :param sequence_scores: (shape: (beam_size, 1)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    candidates = set()
    finished_candidates = set()
    # the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row, col = int(row), int(col)
        seq_score = float(seq_score)
        new_item = hypotheses[row].advance(col)
        cand = ConstrainedCandidate(row, col, seq_score, new_item)
        if hypotheses[row].finished():
            finished_candidates.add(cand)
        elif hypotheses[row].is_valid(col):
            candidates.add(cand)

    hit = np.stack([best_ids, best_word_ids], axis=1).tolist()
    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    best_next = np.argmax(scores, axis=1)
    for row in range(beam_size):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.positive_state.allowed() if hyp.positive_state is not None else set()

        # (3) add the single-best item after this (if it's valid)
        col = int(best_next[row])
        if hyp.is_valid(col):
            nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            if [row, col] not in hit:
                new_item = hyp.advance(col)
                score = scores[row, col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                if hyp.finished():
                    finished_candidates.add(cand)
                elif hyp.is_valid(col):
                    candidates.add(cand)

    if num_fill is not None:
        assert num_fill > beam_size, "at least select number of beam candidates"
    else:
        num_fill = beam_size

    chunk_candidates = []
    if candidates:
        # Sort the candidates.
        sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)
        # prune candidates with low score
        assert prune_factor >= 1, 'prune factor must be greater than or equal to 1'
        # sorted_candidates = sorted_candidates[:int(beam_size * prune_factor)]
        max_satisfy = max([x.hypothesis.num_met() for x in sorted_candidates[:int(beam_size * prune_factor)]])
        sorted_candidates = [x for x in sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]
        sorted_candidates = sorted_candidates[:int(beam_size * prune_factor)]

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_process() for x in sorted_candidates])
        grouped_order_candidates = [[x for x in sorted_candidates if x.hypothesis.met_process() == o] for o in all_orders]

        # Group the top_i candidate of each group in chunk
        chunk_candidates = []
        num_chunk = max([len(x) for x in grouped_order_candidates])
        for i in range(num_chunk):
            chunk_i = []
            for g in grouped_order_candidates:
                if len(g) > i:
                    chunk_i.append(g[i])
            chunk_candidates.append(chunk_i)
        # Sort candidates in each chunk by score
        chunk_candidates = [sorted(x, key=attrgetter('score'), reverse=True) for x in chunk_candidates]

    # TODO: abandon candidates which cannot meet all constraints at max length
    pruned_candidates = sorted(finished_candidates, key=attrgetter('score'), reverse=True)[:beam_size]
    num_finish = len(pruned_candidates)
    for chunk in chunk_candidates:
        if len(pruned_candidates) >= num_fill:
            break

        chunk = [x for x in chunk if x not in pruned_candidates]
        if not chunk:
            continue

        pruned_candidates.extend(chunk[:num_fill - len(pruned_candidates)])

    if num_fill > beam_size:
        select_num = num_finish + beam_size
        complete_candidates = sorted(pruned_candidates[:num_finish], key=attrgetter('score'), reverse=True)
        include_candidates = sorted(pruned_candidates[num_finish:select_num], key=attrgetter('score'), reverse=True)
        extra_candidates = sorted(pruned_candidates[select_num:], key=attrgetter('score'), reverse=True)
        pruned_candidates = complete_candidates + include_candidates + extra_candidates
    else:
        pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'), reverse=True)

    num_pruned_candidates = len(pruned_candidates)

    inactive = np.zeros(num_fill)
    inactive[:num_pruned_candidates] = 0

    # Pad the beam so array assignment still works
    if num_pruned_candidates < num_fill:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (num_fill - num_pruned_candidates)

    assert len(pruned_candidates) == num_fill, 'candidates number mismatch'

    return (sentno,
            np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            [x.hypothesis.num_met() for x in pruned_candidates])
