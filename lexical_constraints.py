import copy
import logging
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import collections

import numpy as np
import torch

logger = logging.getLogger(__name__)

Phrase = List[int]
Literal = Tuple[Phrase, bool]
# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[Phrase]
ClauseConstraintList = List[List[Literal]]


class Trie:
    """
    Represents a set of phrasal constraints for an input sentence.
    These are organized into a trie.
    """
    def __init__(self,
                 raw_phrases: Optional[RawConstraintList] = None,
                 parent_arc: int = None,
                 parent_trie: 'Trie' = None) -> None:
        self.final_ids = set()  # type: Set[int]
        self.children = {}  # type: Dict[int,'Trie']
        self.parent_arc = parent_arc
        self.parent_trie = parent_trie

        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)

    def add_phrase(self,
                   phrase: List[int]) -> None:
        """
        Recursively adds a phrase to this trie node.

        :param phrase: A list of word IDs to add to this trie node.
        """
        if len(phrase) == 1:
            self.final_ids.add(phrase[0])
        else:
            next_word = phrase[0]
            if next_word not in self.children:
                self.children[next_word] = Trie(parent_arc=next_word, parent_trie=self)
            self.step(next_word).add_phrase(phrase[1:])

    def delete_phrase(self,
                      phrase: List[int]) -> None:
        """
        Recursively deletes a phrase to this trie node.

        :param phrase: A list of word IDs to delete in this trie node.
        """
        if len(phrase) == 1:
            assert phrase[0] in self.final_ids, f"Trie {str(self)} \nDo not contain {phrase}"
            self.final_ids.remove(phrase[0])
        else:
            next_word = phrase[0]
            assert next_word in self.children.keys(), f"Trie {str(self)} \nDo not contain {phrase}"
            self.step(next_word).delete_phrase(phrase[1:])

        # Move the arc to an empty node to final_ids of its parent
        for arc in list(self.children):
            if len(self.children[arc]) == 0:
                self.children.pop(arc)

    def check_phrase(self,
                     phrase: List[int]) -> bool:
        """
        Check whether a phrase is in this trie.

        :param phrase: A list of word IDs to check existence.
        """
        if len(phrase) == 1:
            return phrase[0] in self.final_ids
        else:
            next_word = phrase[0]
            if next_word in self.children:
                return self.step(next_word).check_phrase(phrase[1:])
            return False

    def trace_phrase(self,
                     word_id: int) -> List[int]:
        """
        Recursively backward to get word ids in a phrase.

        :param word_id: The last word IDs in phrase.
        """
        assert word_id in self.final_ids, f"{word_id} does not in trie node {self.final_ids}"
        phrase = self.trace_arcs()
        phrase.append(word_id)
        return phrase

    def trace_arcs(self,) -> List[int]:
        """
        Recursively backward to get arc to ancestor
        """
        arcs = []
        parent_trie, parent_arc = self.parent_trie, self.parent_arc
        while parent_trie is not None:
            arcs.append(parent_arc)
            parent_arc = parent_trie.parent_arc
            parent_trie = parent_trie.parent_trie
        arcs.reverse()
        return arcs

    def __str__(self) -> str:
        s = f'({list(self.final_ids)}'
        for child_id in self.children.keys():
            s += f' -> {child_id} {self.children[child_id]}'
        s += ')'
        return s

    def __len__(self) -> int:
        """
        Returns the number of phrases represented in the trie.
        """
        phrase_count = len(self.final_ids)
        for child in self.children.values():
            phrase_count += len(child)
        return phrase_count

    def step(self, word_id: int) -> Optional['Trie']:
        """
        Returns the child node along the requested arc.

        :param word_id: requested arc.
        :return: The child node along the requested arc, or None if no such arc exists.
        """
        return self.children.get(word_id, None)

    def descend(self,
              arcs: List[int]) -> Optional['Trie']:
        pointer = self
        for arc in arcs:
            if pointer is None:
                break
            pointer = pointer.step(word_id=arc)
        return pointer

    def final(self) -> Set[int]:
        """
        Returns the set of final ids at this node.

        :return: The set of word IDs that end a constraint at this state.
        """
        return self.final_ids


class NegativeState:
    """
    Represents the state of a hypothesis in the AvoidTrie.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param avoid_trie: The trie containing the phrases to avoid.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 avoid_trie: Trie,
                 state: List[Trie] = None) -> None:

        self.root = avoid_trie
        self.state = state if state else [self.root]

    def consume(self, word_id: int) -> 'NegativeState':
        """
        Consumes a word, and updates the state based on it. Returns new objects on a state change.

        The next state for a word can be tricky. Here are the cases:
        (1) If the word is found in our set of outgoing child arcs, we take that transition.
        (2) If the word is not found, and we are not in the root state, we need to reset.
            This means we pretend we were in the root state, and see if we can take a step
        (3) Otherwise, if we are not already in the root state (i.e., we were partially through
            the trie), we need to create a new object whose state is the root state
        (4) Finally, if we couldn't advance and were already in the root state, we can reuse
            this object.

        :param word_id: The word that was just generated.
        """
        new_state = []
        for state in set(self.state + [self.root]):
            if word_id in state.children:
                new_state.append(state.step(word_id))

        if new_state:
            return NegativeState(self.root, new_state)
        else:
            if len(self.state) == 1 and self.root == self.state[0]:
                return self
            else:
                return NegativeState(self.root, [self.root])

    def avoid(self) -> Set[int]:
        """
        Returns a set of word IDs that should be avoided. This includes the set of final states from the
        root node, which are single tokens that must never be generated.

        :return: A set of integers representing words that must not be generated next by this hypothesis.
        """
        return self.root.final().union(*[state.final() for state in self.state])

    def __str__(self) -> str:
        return str(self.state)


class NegativeBatch:
    """
    Represents a set of phrasal constraints for all items in the batch.
    For each hypotheses, there is an AvoidTrie tracking its state.

    :param beam_size: The beam size.
    :param avoid_list: The list of lists (raw phrasal constraints as IDs, one for each item in the batch).
    """
    def __init__(self,
                 beam_size: int,
                 avoid_list: Optional[List[RawConstraintList]] = None) -> None:

        self.avoid_states = []  # type: List[NegativeState]

        # Store the sentence-level tries for each item in their portions of the beam
        if avoid_list is not None:
            for literal_phrases in avoid_list:
                self.avoid_states += [NegativeState(Trie(literal_phrases))] * beam_size

    def reorder(self, indices: torch.Tensor) -> None:
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.avoid_states:
            self.avoid_states = [self.avoid_states[x] for x in indices.numpy()]

    def consume(self, word_ids: torch.Tensor) -> None:
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.numpy().tolist()
        for i, word_id in enumerate(word_ids):
            if self.avoid_states:
                self.avoid_states[i] = self.avoid_states[i].consume(word_id)

    def avoid(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Assembles a list of per-hypothesis words to avoid. The indices are (x, y) pairs into the scores
        array, which has dimensions (beam_size, target_vocab_size). These values are then used by the caller
        to set these items to np.inf so they won't be selected. Words to be avoided are selected by
        consulting both the global trie of phrases and the sentence-specific one.

        :return: Two lists of indices: the x coordinates and y coordinates.
        """
        to_avoid = set()  # type: Set[Tuple[int, int]]
        for i, state in enumerate(self.avoid_states):
            for word_id in state.avoid():
                to_avoid.add((i, word_id))

        return tuple(zip(*to_avoid))  # type: ignore


class PositiveState:
    """
    Represents a set of words and phrases that must appear in the output.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param positive_trie: The trie containing the phrases to appear.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 positive_trie: Trie,
                 state: List[Trie] = None,
                 met_phrases: RawConstraintList = None) -> None:

        self.root = positive_trie
        self.state = state if state else [self.root]
        self.met_phrases = met_phrases

    def __str__(self):
        s = f'Root: {self.root}\nState: ['
        for state in self.state:
            s += f'{state}, '
        s += f']\nMet_phrases: {self.met_phrases}'
        return s

    def allowed(self) -> Set[int]:
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is the list of all unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """
        allow = self.root.final().union(*[state.final() for state in self.state])
        allow |= set(self.root.children.keys()).union(*[set(state.children.keys()) for state in self.state])
        return allow

    def advance(self, word_id: int) -> 'PositiveState':
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to allow constraints
        to be added as unconstrained words, so if the next word is
        invalid, we must "back out" of the current (incomplete) phrase,
        re-setting all of its words as unmet.

        :param word_id: The word ID to advance on.
        :return: A deep copy of the object, advanced on word_id.
        """
        new_state, met_phrases = [], []
        for state in set(self.state + [self.root]):
            if word_id in state.children:
                new_state.append(state.step(word_id))
            if word_id in state.final_ids:
                met_phrases.append(state.trace_phrase(word_id))

        if new_state:
            return PositiveState(self.root, new_state, met_phrases if met_phrases else None)
        else:
            if len(self.state) == 1 and self.root == self.state[0] and not met_phrases:
                return self
            else:
                return PositiveState(self.root, [self.root], met_phrases if met_phrases else None)


class Clause:
    """
    Object used to hold clause.

    :param idx: The id of this clause.
    :param positive: The positive constraints in this clause.
    :param negative: The soft negative constraints in this clause.
    :param satisfy: whether this clause is satisfied
    """

    __slots__ = ('idx', 'positive', 'negative', 'satisfy')

    def __init__(self,
                 idx: int,
                 positive: List[Phrase],
                 negative: List[Phrase],
                 satisfy: float) -> None:
        self.idx = idx
        self.positive = positive
        self.negative = negative
        self.satisfy = satisfy

    def __str__(self):
        return f'clause(id={self.idx}, positive={self.positive}, negative={self.negative}, satisfy={self.satisfy})'


def is_prefix(pref: List[int],
              phrase: List[int]):
    if not pref:
        return False
    return pref == phrase[:len(pref)]


class ConstrainedHypothesis:
    """
    Keep track of positive and negative constraint

    hard negative constraint will not be generated in any cases
    soft negative constraint might be generated in some case due to OR gate in clause
    positive constraints will be encourage to appear

    :param constraint_list: A list of clause constraints (each represented as a list of literals).
    """
    def __init__(self,
                 constraint_list: ClauseConstraintList,
                 eos_id: Union[int, list]
                 ) -> None:
        self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
        self.clauses = []  # type: List[Clause]

        hard_neg_pool, soft_neg_pool, pos_pool = [], [], []  # type: RawConstraintList
        for idx, clause in enumerate(constraint_list):
            if not clause:
                continue
            pos_phrases, neg_phrases = [l[0] for l in clause if l[1]], [l[0] for l in clause if not l[1]]
            # clause contains single negative literal
            if not pos_phrases and len(neg_phrases) == 1:
                hard_neg_pool.extend(neg_phrases)
                #self.clauses.append(Clause(idx=idx, positive=[], negative=neg_phrases, satisfy=True))
            # clause contains multiple negative literals or both negative and positive literals
            elif neg_phrases:
                soft_neg_pool.extend(neg_phrases)
                self.clauses.append(Clause(idx=idx, positive=pos_phrases, negative=neg_phrases, satisfy=True))
            # clause contains only positive literals
            elif pos_phrases and not neg_phrases:
                pos_pool.extend(pos_phrases)
                self.clauses.append(Clause(idx=idx, positive=pos_phrases, negative=[], satisfy=False))
            else:
                import ipdb
                ipdb.set_trace()
                raise ValueError(f'Invalid state {clause}, should not be reached')

        self.hard_negative_state = NegativeState(Trie(hard_neg_pool)) if hard_neg_pool else None
        self.soft_negative_state = NegativeState(Trie(soft_neg_pool)) if soft_neg_pool else None
        self.positive_state = PositiveState(Trie(pos_pool)) if pos_pool else None

        self.orders = []
        self.in_process = None
        self.max_process = 0

    def __len__(self) -> int:
        """
        :return: The number of constraints.
        """
        return len(self.clauses)

    def __str__(self) -> str:
        return '\n'.join([str(c) for c in self.clauses])

    def size(self) -> int:
        """
        :return: the number of constraints
        """
        return len(self.clauses)

    def num_met(self) -> int:
        """
        :return: the number of constraints that have been met.
        """
        if not self.clauses:
            return 0
        return sum([int(c.satisfy) for c in self.clauses])

    def met_order(self) -> tuple:
        """
        :return: the number of constraints that have been met.
        """
        return tuple(sorted(self.orders))

    def clause_in_process(self) -> tuple:
        """
        :return: the index of clause that's in generation.
        """
        return tuple(self.in_process)

    def num_needed(self) -> int:
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def finished(self) -> bool:
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid: int) -> bool:
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid not in self.eos_id

    def avoid(self) -> Set[int]:
        banned = self.hard_negative_state.avoid() if self.hard_negative_state is not None else set()
        return banned

    def eos(self) -> list:
        """
        :return: Return EOS id.
        """
        return self.eos_id

    def advance(self, word_id: int) -> 'ConstrainedHypothesis':
        """
        Updates the constraints object based on advancing on word_id.
        If one of literals in a clause is satisfied, we mark this clause as satisfied

        :param word_id: The word ID to advance on.
        """
        obj = copy.deepcopy(self)

        if obj.soft_negative_state is not None:
            raise NotImplementedError

        if obj.hard_negative_state is not None:
            obj.hard_negative_state = obj.hard_negative_state.consume(word_id)

        if obj.positive_state is not None:
            temp_pos_state = obj.positive_state.advance(word_id)
            if temp_pos_state.met_phrases is not None:
                # get newly satisfied positive literals
                phrases_to_delete = []
                newly_met_clause = set()
                for phrase in temp_pos_state.met_phrases:
                    for clause in obj.clauses:
                        if not clause.satisfy and phrase in clause.positive:
                            phrases_to_delete.extend(clause.positive)
                            clause.satisfy = True
                            assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                            newly_met_clause.add(clause.idx)
                obj.orders.extend(sorted(newly_met_clause))

                # delete newly satisfied literals from positive trie state
                new_root = copy.deepcopy(temp_pos_state.root)
                phrases_to_delete = [list(i) for i in set(map(tuple, phrases_to_delete))]
                for phrase in phrases_to_delete:
                    if new_root.check_phrase(phrase):
                        new_root.delete_phrase(phrase)
                new_trie_states = set()
                for state in temp_pos_state.state:
                    # pointer at root state
                    if state.parent_trie is None:
                        new_trie_states.add(new_root)
                    else:
                        trace = state.trace_arcs()
                        new_state = new_root.descend(trace)
                        if new_state is not None:
                            new_trie_states.add(new_state)
                obj.positive_state = PositiveState(positive_trie=new_root, state=list(new_trie_states))
            else:
                obj.positive_state = temp_pos_state

            history = [s.trace_arcs() for s in obj.positive_state.state]
            newly_in_process = set()
            max_process = 0
            for phrase in history:
                for clause in obj.clauses:
                    phrase_in_process = [c for c in clause.positive if is_prefix(phrase, c)]
                    if not clause.satisfy and bool(phrase_in_process):
                        process_portion = len(phrase) / min([len(x) for x in phrase_in_process])
                        max_process = max(max_process, process_portion)
                        assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                        newly_in_process.add(clause.idx)
            obj.in_process = sorted(newly_in_process)
            obj.max_process = max_process
        return obj


def init_batch(raw_constraints: List[ClauseConstraintList],
               beam_size: int,
               eos_id: Union[int, list]) -> List[Optional[ConstrainedHypothesis]]:
    """
    :param raw_constraints: The list of clause constraints.
    :param beam_size: The beam size.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints_list = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
    for i, raw_list in enumerate(raw_constraints):
        hyp = ConstrainedHypothesis(raw_list, eos_id)
        idx = i * beam_size
        constraints_list[idx:idx + beam_size] = [copy.deepcopy(hyp) for _ in range(beam_size)]
    return constraints_list


class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis', 'rank')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis,
                 rank: float = None,) -> None:
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis
        self.rank = rank

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


if __name__ == '__main__':
    clauses = [[[([3, 4, 5], True), ([3, 4], True), ([4, 5], True)], [([3, 4], True), ([6], True), ([7], True)]],
               [[([6], True), ([6, 7], True), ([6, 7, 8], True)], [([6, 9], True), ([6, 4, 9], True)]],
               [[([3, 4, 5], True)], [([3, 4], True)], [([4, 5], True)]],
               [[([3, 4], True)], [([2, 3, 5], True)], [([6, 5], True)]]]

    constraints = init_batch(raw_constraints=clauses,
                             beam_size=1,
                             eos_id=0)

    constraint = constraints[2]
    print(constraint)
    print(constraints)
    print()
    for w in [2, 3, 4, 5]:
        constraint = constraint.advance(w)
        print(constraint)
        print(constraint.positive_state)
        print(constraint.positive_state.allowed())
        print(constraint.met_order())
        print(constraint.clause_in_process())
        print()


