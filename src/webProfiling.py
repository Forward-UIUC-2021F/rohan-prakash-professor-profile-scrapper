'''
Models used are:
----------------
* Tree-related Conditional Random field (TCRF) -- for Tagging an observation x, which is a given homepage

* Tree-based Reparameterization (TRP) --  compute the approximate probabilities of the factors 
______________________________________________________________
Using the tags, we can perform extraction of 16 profile properties, which cover 95.71% of the property values on
the Web pages).
'''

import torch
from torch import nn


# class TCRF(nn.Module):
#     '''
#     Tree-related Conditional Random Field (TCRF).

#     Args:
#         nb_labels (int): number of labels in your tagset, including special symbols.
#         bos_tag_id (int): integer representing the beginning of sentence symbol in
#             your tagset.
#         eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
#         batch_first (bool): Whether the first dimension represents the batch dimension.
#     '''
#     def __init__(self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True) -> None:
#         super().__init__()

#         self.nb_labels = nb_labels
#         self.BOS_TAG_ID = bos_tag_id
#         self.EOS_TAG_ID = eos_tag_id
#         self.batch_first = batch_first

#         self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
#         self.init_weights()

#     def init_weights(self):
#         pass



class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

#############################################################
# SETUP
    def __init__(
        self, nb_labels, bos_tag_id, eos_tag_id, batch_first=True
    ):
        super().__init__()

        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from a random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # enforce contraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero

        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        # no transition alloed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

#############################################################
# DEFINE LOSS FUNCTION
    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
        Returns:
            torch.Tensor: the (summed) log-likelihoods of each sequence in the batch.
                Shape of (1,)
        """

        # fix tensors order by setting batch as the first dimension
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)


#############################################################
# COMPUTE SCORES
    def _compute_scores(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from BOS to the first tags for each batch
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):

            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        scores += self.transitions[last_tags, self.EOS_TAG_ID]

        return scores


#############################################################
# COMPUTE Z(x), Log Partition function determined during training
    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            alpha_t = []

            for tag in range(nb_labels):

                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels
                # since it will be the same for all previous tags
                # (bs, nb_labels)
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transitions[:, tag]

                # broadcast the transition scores to all batches
                # (bs, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)

#############################################################
# DECODE SEQUENCE for MAX Scores
    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):

                # get the emission for the current tag and broadcast to all labels
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag and broadcast to all batches
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # so far is exactly like the forward algorithm,
                # but now, instead of calculating the logsumexp,
                # we will find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score_tag for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

#############################################################
# FIND BEST SEQUENCE OF LABELs

