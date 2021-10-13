import numpy as np
import math
from scipy.stats import rankdata
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union

from lexical_constraints import ConstrainedHypothesis, ConstrainedCandidate

def topk(timestep: int,
         batch_size: int,
         beam_size: int,
         prune_factor: int,
         sat_tolerance: int,
         beta: float,
         early_stop: float,
         inactive: np.array,
         scores: np.array,
         hypotheses: List[ConstrainedHypothesis],
         num_mets: List[int],
         best_ids: np.array,
         best_word_ids: np.array,
         seq_scores: np.array) -> Tuple[np.array, np.array, np.array, List[ConstrainedHypothesis], List[int]]:
    """
    Builds a new topk list such that the beam contains hypotheses having completed different numbers of constraints.
    These items are built from three different types: (1) the best items across the whole
    scores matrix, (2) the set of words that must follow existing constraints, and (3) k-best items from each row.

    :param batch_size: The number of segments in the batch.
    :param beam_size: The length of the beam for each segment.
    :param prune_factor:
    :param sat_tolerance:
    :param inactive: Array listing inactive rows (shape: (batch_size, beam_size,)).
    :param scores: The scores array (shape: batch_size, beam_size, target_vocab_size)).
    :param hypotheses: The list of hypothesis objects. (length: (batch_size * beam_size,))
    :param num_mets: The list of int how many constraints satisfied. (length: (batch_size * beam_size))
    :param best_ids: The current list of best hypotheses (shape: (batch_size, beam_size)).
    :param best_word_ids: The parallel list of best word IDs (shape: (batch_size, beam_size)).
    :param seq_scores: (shape: (batch_size, beam_size)).
    :return: A tuple containing the best hypothesis rows, the best hypothesis words, the scores,
        the updated constrained hypotheses, and the updated set of inactive hypotheses.
    """

    for sentno in range(batch_size):
        rows = slice(sentno * beam_size, sentno * beam_size + beam_size)
        if hypotheses[rows.start].size() > 0:
            best_ids[sentno], best_word_ids[sentno], seq_scores[sentno],\
                                 hypotheses[rows], num_mets[rows] = _sequential_topk(timestep,
                                                                                     beam_size,
                                                                                     prune_factor,
                                                                                     sat_tolerance,
                                                                                     beta,
                                                                                     early_stop,
                                                                                     inactive[sentno],
                                                                                     scores[sentno],
                                                                                     hypotheses[rows],
                                                                                     best_ids[sentno],
                                                                                     best_word_ids[sentno],
                                                                                     seq_scores[sentno])
    return best_ids, best_word_ids, seq_scores, hypotheses, num_mets


def _sequential_topk(timestep: int,
                     beam_size: int,
                     prune_factor: int,
                     sat_tolerance: int,
                     beta: float,
                     early_stop: float,
                     inactive: np.array,
                     scores: np.array,
                     hypotheses: List[ConstrainedHypothesis],
                     best_ids: np.array,
                     best_word_ids: np.array,
                     sequence_scores: np.array) -> Tuple[np.array, np.array, np.array,
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
    rank = rankdata(-1 * scores, method='dense').reshape(scores.shape)

    # (1) Add all of the top-k items (which were passed) in as long as they pass the constraints
    for row, col, seq_score in zip(best_ids, best_word_ids, sequence_scores):
        row, col = int(row), int(col)
        seq_score = float(seq_score)
        new_item = hypotheses[row].advance(col)
        cand = ConstrainedCandidate(row, col, seq_score, new_item)
        if hypotheses[row].finished():
            finished_candidates.add(cand)
        elif hypotheses[row].is_valid(col) or int(best_next[row]) == col:
            candidates.add(cand)

    hit = np.stack([best_ids, best_word_ids], axis=1).tolist()
    # For each hypothesis, we add (2) all the constraints that could follow it and
    # (3) the best item (constrained or not) in that row
    for row in range(beam_size if timestep > 0 else 1):
        if inactive[row]:
            continue

        hyp = hypotheses[row]

        # (2) add all the constraints that could extend this
        nextones = hyp.positive_state.allowed()

        # (3) add the best items (if it's valid)
        best_k = np.argsort(scores[row])[::-1][:beam_size]
        for col in best_k:
            if hyp.is_valid(col):
                nextones.add(col)

        # Now, create new candidates for each of these items
        for col in nextones:
            if [row, col] not in hit and rank[row, col] < prune_factor:
                new_item = hyp.advance(col)
                score = scores[row, col]
                cand = ConstrainedCandidate(row, col, score, new_item)
                if hyp.finished() and col in hyp.eos():
                    finished_candidates.add(cand)
                else:
                    candidates.add(cand)

        # Add finished candidates in finished set:
        if hyp.finished() and early_stop is not None:
            best_k = np.argsort(scores[row])[::-1][:int(beam_size * early_stop)]
            for col in best_k:
                if col in hyp.eos():
                    new_item = hyp.advance(col)
                    score = scores[row, col]
                    cand = ConstrainedCandidate(row, col, score, new_item)
                    finished_candidates.add(cand)

    chunk_candidates = []
    if candidates:
        # Sort the candidates.
        sorted_candidates = sorted(candidates, key=attrgetter('score'), reverse=True)
        max_satisfy = max([x.hypothesis.num_met() for x in sorted_candidates])
        sorted_candidates = [x for x in sorted_candidates if x.hypothesis.num_met() >= max_satisfy - sat_tolerance]

        for cand in sorted_candidates:
            cand.rank = cand.score / (timestep + 1)
            if cand.hypothesis.max_process:
                cand.rank -= beta * math.log(cand.hypothesis.max_process)
        sorted_candidates = sorted(sorted_candidates, key=attrgetter('rank'), reverse=True)

        # Bucket candidates in each group by met order
        all_orders = set([x.hypothesis.met_order() for x in sorted_candidates])
        grouped_order_candidates = [[x for x in sorted_candidates if x.hypothesis.met_order() == o] for o in all_orders]

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
        chunk_candidates = [sorted(x, key=attrgetter('rank'), reverse=True) for x in chunk_candidates]

    pruned_candidates = sorted(finished_candidates, key=attrgetter('score'), reverse=True)[:beam_size]
    for chunk in chunk_candidates:
        if len(pruned_candidates) >= beam_size:
            break

        chunk = [x for x in chunk if x not in pruned_candidates]
        if not chunk:
            continue

        pruned_candidates.extend(chunk[:beam_size - len(pruned_candidates)])
    pruned_candidates = sorted(pruned_candidates, key=attrgetter('score'), reverse=True)
    num_pruned_candidates = len(pruned_candidates)

    inactive[:num_pruned_candidates] = 0

    # Pad the beam so array assignment still works
    if num_pruned_candidates < beam_size:
        inactive[num_pruned_candidates:] = 1
        pruned_candidates += [pruned_candidates[num_pruned_candidates - 1]] * (beam_size - num_pruned_candidates)
    return (np.array([x.row for x in pruned_candidates]),
            np.array([x.col for x in pruned_candidates]),
            np.array([x.score for x in pruned_candidates]),
            [x.hypothesis for x in pruned_candidates],
            [x.hypothesis.num_met() for x in pruned_candidates])
