import torch
import math
import numpy as np

from transformers import UnilmForSeq2SeqDecode
from unilm.topK import topk


class Candidate:
    """
    Object used to hold candidates for the beam .

    :param frame_id: The row in the scores matrix.
    :param pos_in_frame: The position of eos in backpointers
    :param score: the associated accumulated score.
    :param num_met: number of constraints met
    """

    __slots__ = ('frame_id', 'pos_in_frame', 'score', 'num_met')

    def __init__(self,
                 frame_id: int,
                 pos_in_frame: int,
                 score: float,
                 num_met: int) -> None:
        self.frame_id = frame_id
        self.pos_in_frame = pos_in_frame
        self.score = score
        self.num_met = num_met

    def __hash__(self):
        return hash((self.frame_id, self.pos_in_frame))

    def __eq__(self, other):
        return self.frame_id == other.frame_id and self.pos_in_frame == other.pos_in_frame

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.frame_id, self.pos_in_frame, self.score, self.num_met)


class UnilmConstrainDecode(UnilmForSeq2SeqDecode):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0,
                 prune_factor=50, sat_tolerance=2, beta=0., early_stop=1.5):
        super(UnilmConstrainDecode, self).__init__(config, mask_word_id, search_beam_size, length_penalty, eos_id,
                                                   sos_id, forbid_duplicate_ngrams, forbid_ignore_set, ngram_size,
                                                   min_len)
        self.prune_factor = prune_factor
        self.sat_tolerance = sat_tolerance
        self.beta = beta
        self.early_stop = early_stop
        self.ngram_size = 2

    def decode(self, input_ids, token_type_ids, position_ids, attention_mask, constraints):
        if self.search_beam_size < 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        num_mets = [x.num_met() for x in constraints]
        k_num_mets = None

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        step_num_mets = []
        forbid_word_mask = None
        buf_matrix = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                  start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.cls(last_hidden)
            # (batch_size * K, 1, vocab_size)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            # (batch_size * K, 1, vocab_size)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                full_scores = torch.reshape(log_scores, [batch_size, 1, -1])
                full_scores = full_scores.repeat(*[1, K, 1])
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                previous_scores = last_eos * (-10000.0) + last_seq_scores
                # (batch_size * K, 1, vocab_size)
                full_scores = log_scores + previous_scores
                full_scores = torch.reshape(full_scores, [batch_size, K, -1])
                # (batch_size * K, 1, K)
                kk_scores += previous_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                # (batch_size, K)
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)

            if constraints is not None:
                inactive = beam_masks[-1] if total_scores else torch.zeros(batch_size, K)
                best_ids, best_word_ids, seq_scores, hypotheses, num_mets = topk(timestep=len(step_back_ptrs),
                                                                                 batch_size=batch_size,
                                                                                 beam_size=K,
                                                                                 prune_factor=self.prune_factor,
                                                                                 sat_tolerance=self.sat_tolerance,
                                                                                 beta=self.beta,
                                                                                 early_stop=self.early_stop,
                                                                                 inactive=inactive.cpu().numpy(),
                                                                                 scores=full_scores.cpu().numpy(),
                                                                                 hypotheses=constraints,
                                                                                 num_mets=num_mets,
                                                                                 best_ids=back_ptrs.cpu().numpy(),
                                                                                 best_word_ids=k_ids.cpu().numpy(),
                                                                                 seq_scores=k_scores.cpu().numpy())

                back_ptrs = torch.tensor(best_ids, dtype=back_ptrs.dtype, device=back_ptrs.device)
                k_ids = torch.tensor(best_word_ids, dtype=k_ids.dtype, device=k_ids.device)
                k_scores = torch.tensor(seq_scores, dtype=k_scores.dtype, device=k_scores.device)
                constraints = hypotheses
                k_num_mets = torch.reshape(torch.tensor(num_mets, device=k_scores.device), k_scores.shape)

            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)
            step_num_mets.append(k_num_mets)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :])
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(
                    prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        step_num_mets = [x.tolist() for x in step_num_mets]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': [], 'mets': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            mets = [x[b] for x in step_num_mets]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            traces['mets'].append(mets)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break

            candidates = []
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        nm = mets[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        candidates.append(Candidate(frame_id=fid, pos_in_frame=i, score=s, num_met=nm))
            if len(candidates) == 0:
                traces['pred_seq'].append([0])
            else:
                candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:5]
                candidates = sorted(candidates, key=lambda x: (x.num_met, x.score), reverse=True)

                def get_seq(frame_id, pos_in_frame):
                    seq = [wids_list[frame_id][pos_in_frame]]
                    for fid in range(frame_id, 0, -1):
                        pos_in_frame = ptrs[fid][pos_in_frame]
                        seq.append(wids_list[fid - 1][pos_in_frame])
                    seq.reverse()
                    return seq

                traces['pred_seq'].append(get_seq(candidates[0].frame_id, candidates[0].pos_in_frame))

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs', 'mets'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces