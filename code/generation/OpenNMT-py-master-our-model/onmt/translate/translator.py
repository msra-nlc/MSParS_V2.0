#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt)

    translator = Translator(model, fields, opt, model_opt,
                            global_scorer=scorer, out_file=out_file,
                            report_score=report_score, logger=logger)
    translator1 = Translator1(model, fields, opt, model_opt,
                            global_scorer=scorer, out_file=out_file, out_file1=codecs.open("pred_entity1.txt", 'w', 'utf-8'),
                            report_score=report_score, logger=logger)
    translator2 = Translator2(model, fields, opt, model_opt,
                             global_scorer=scorer, out_file=out_file, out_file1=codecs.open("pred_entity2.txt","w","utf-8"),
                             report_score=report_score, logger=logger)

    return translator, translator1, translator2


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 opt,
                 model_opt,
                 global_scorer=None,
                 out_file=None,
                 report_score=True,
                 logger=None):

        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)
        self.sample_rate = opt.sample_rate
        self.window_size = opt.window_size
        self.window_stride = opt.window_stride
        self.window = opt.window
        self.image_channel_size = opt.image_channel_size
        self.replace_unk = opt.replace_unk
        self.data_type = opt.data_type
        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.fast = opt.fast

        self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        #self.out_file1 = out_file1
        #self.out_file2 = out_file2
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_data_iter=src_data_iter,
                          src_seq_length_trunc=200,
                          tgt_path=tgt_path,
                          tgt_data_iter=tgt_data_iter,
                          src_dir=src_dir,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=self.use_filter_pred,
                          image_channel_size=self.image_channel_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=self.fast)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                if len(n_best_preds[0]) == 0:
                    n_best_preds[0] = "unknown"
                self.out_file.write(n_best_preds[0] +" <TSP> unknown" + " <TSP> unknown" + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))


        return all_scores, all_predictions

    def translate_batch(self, batch, data, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():

            return self._translate_batch(batch, data)

    def _run_encoder(self, batch, data_type):
        src = inputters.make_features(batch, 'src', data_type)
        src1 = inputters.make_features(batch, 'src1', data_type)
        src_lengths = None
        src_lengths1 = None
        if data_type == 'text':
            _, src_lengths = batch.src
            _, src_lengths1 = batch.src1
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        enc_states1, memory_bank1, src_lengths1 = self.model.encoder1(
            src1, src_lengths1)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))

        if src_lengths1 is None:
            assert not isinstance(memory_bank1, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths1 = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank1) \
                .long() \
                .fill_(memory_bank1.size(0))
        return src, enc_states, memory_bank, src_lengths,src1, enc_states1, memory_bank1, src_lengths1

    def _decode_and_generate(self, decoder_input, memory_bank, batch, data,
                             memory_lengths, src_map=None,
                             step=None, batch_offset=None):

        if self.copy_attn:
            # Turn any copied words to UNKs (index 0).
            decoder_input = decoder_input.masked_fill(
                decoder_input.gt(len(self.fields["tgt"].vocab) - 1), 0)

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=step)

        # Generator forward.
        if not self.copy_attn:
            attn = dec_attn["std"]
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = data.collapse_copy_scores(
                scores,
                batch,
                self.fields["tgt"].vocab,
                data.src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset)
            scores = scores.view(decoder_input.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn



    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths,src1, enc_states1, memory_bank1, src_lengths1 = self._run_encoder(
            batch, data_type)
        self.model.decoder.init_state(src, memory_bank, enc_states)


        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank, src_lengths, data, batch.src_map
                if data_type == 'text' and self.copy_attn else None)
            self.model.decoder.init_state(
                src, memory_bank, enc_states, with_cache=True)
        else:
            results["gold_score"] = [0] * batch_size

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if data.data_type == 'text' and self.copy_attn else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.get_current_state() for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = \
                self._decode_and_generate(inp, memory_bank, batch, data,
                                          memory_lengths=memory_lengths,
                                          src_map=src_map, step=i)

            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths[j]])
                select_indices_array.append(
                    b.get_current_origin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths, data, src_map):
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]

        log_probs, attn = \
            self._decode_and_generate(tgt_in, memory_bank, batch, data,
                                      memory_lengths=src_lengths,
                                      src_map=src_map)
        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]

        log_probs[:, :, tgt_pad] = 0
        gold = batch.tgt[1:].unsqueeze(2)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores



class Translator1(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 opt,
                 model_opt,
                 global_scorer=None,
                 out_file=None,out_file1 = None,
                 report_score=True,
                 logger=None):

        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)
        self.sample_rate = opt.sample_rate
        self.window_size = opt.window_size
        self.window_stride = opt.window_stride
        self.window = opt.window
        self.image_channel_size = opt.image_channel_size
        self.replace_unk = opt.replace_unk
        self.data_type = opt.data_type
        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.fast = opt.fast

        self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.out_file1 = out_file1
        # self.out_file2 = out_file2
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_seq_length_trunc = 200,
                          src_data_iter=None,
                          tgt_path="transfer-pred.txt",
                          tgt_data_iter=None,
                          src_dir=src_dir,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=self.use_filter_pred,
                          image_channel_size=self.image_channel_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder1 = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, None)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=self.fast)
            #print(batch_data.size())
            translations = builder1.from_batch(batch_data,tag = "src_vocabs1")

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                if len(n_best_preds[0]) == 0:
                    n_best_preds[0] = "unknown"
                self.out_file1.write(n_best_preds[0] + '\n')
                self.out_file1.flush()



        return all_scores, all_predictions

    def translate_batch(self, batch, data, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():

            return self._translate_batch(batch, data)

    def _run_encoder(self, batch, data_type):
        src = inputters.make_features(batch, 'src', data_type)
        src1 = inputters.make_features(batch, 'src1', data_type)
        src_lengths = None
        src_lengths1 = None
        if data_type == 'text':
            _, src_lengths = batch.src
            _, src_lengths1 = batch.src1
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        enc_states1, memory_bank1, src_lengths1 = self.model.encoder1(
            src1, src_lengths1)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank) \
                .long() \
                .fill_(memory_bank.size(0))

        if src_lengths1 is None:
            assert not isinstance(memory_bank1, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths1 = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank1) \
                .long() \
                .fill_(memory_bank1.size(0))
        return src, enc_states, memory_bank, src_lengths, src1, enc_states1, memory_bank1, src_lengths1

    def _decode_and_generate(self, decoder_input, memory_bank, batch, data,
                             memory_lengths, src_map=None,
                             step=None, batch_offset=None):

        if self.copy_attn:
            # Turn any copied words to UNKs (index 0).
            decoder_input = decoder_input.masked_fill(
                decoder_input.gt(len(self.fields["tgt"].vocab) - 1), 0)

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder1(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=step)

        # Generator forward.
        if not self.copy_attn:
            attn = dec_attn["std"]
            log_probs = self.model.generator1(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator1(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = data.collapse_copy_scores(
                scores,
                batch,
                self.fields["tgt"].vocab,
                data.src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset)
            scores = scores.view(decoder_input.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn



    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths, src1, enc_states1, memory_bank1, src_lengths1 = self._run_encoder(
            batch, data_type)
        self.model.decoder.init_state(src, memory_bank, enc_states)
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]
        #print(tgt_in.size())
        dec_out, attns = self.model.decoder(tgt_in, memory_bank,
                                      memory_lengths=src_lengths)

        inin_state_for_decoder_1 = [dec_out[i] for i in batch.tgt1_index]
        self.model.decoder1.init_state(src1, memory_bank1, inin_state_for_decoder_1)
        #dec_out, attns = self.decoder_1(batch.tgt1, memory_bank1,
        #                                    memory_lengths=src_lengths1)

        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []
        results["batch"] = batch
        if "tgt1" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank1, src_lengths1, data, batch.src1_map
                if data_type == 'text' and self.copy_attn else None)
            self.model.decoder1.init_state(
                src1, memory_bank1, inin_state_for_decoder_1, with_cache=True)
        else:
            results["gold_score"] = [0] * batch_size

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map1 = (tile(batch.src1_map, beam_size, dim=1)
                   if data.data_type == 'text' and self.copy_attn else None)
        self.model.decoder1.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank1, tuple):
            memory_bank1 = tuple(tile(x, beam_size, dim=1) for x in memory_bank1)
        else:
            memory_bank1 = tile(memory_bank1, beam_size, dim=1)
        memory_lengths1 = tile(src_lengths1, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(10):
            if all((b.done() for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.get_current_state() for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = \
                self._decode_and_generate(inp, memory_bank1, batch, data,
                                          memory_lengths=memory_lengths1,
                                          src_map=src_map1, step=i)

            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths1[j]])
                select_indices_array.append(
                    b.get_current_origin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder1.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths, data, src_map):
        tgt_in = inputters.make_features(batch, 'tgt1')[:-1]

        log_probs, attn = \
            self._decode_and_generate(tgt_in, memory_bank, batch, data,
                                      memory_lengths=src_lengths,
                                      src_map=src_map)
        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]

        log_probs[:, :, tgt_pad] = 0
        gold = batch.tgt1[1:].unsqueeze(2)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores




class Translator2(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 opt,
                 model_opt,
                 global_scorer=None,
                 out_file=None, out_file1=None,
                 report_score=True,
                 logger=None):

        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)
        self.sample_rate = opt.sample_rate
        self.window_size = opt.window_size
        self.window_stride = opt.window_stride
        self.window = opt.window
        self.image_channel_size = opt.image_channel_size
        self.replace_unk = opt.replace_unk
        self.data_type = opt.data_type
        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.fast = opt.fast

        self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.out_file1 = out_file1
        # self.out_file2 = out_file2
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_data_iter=None,
                          src_seq_length_trunc =200,
                          tgt_path="transfer-pred.txt",
                          tgt_data_iter=None,
                          src_dir=src_dir,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=self.use_filter_pred,
                          image_channel_size=self.image_channel_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)


        builder1 = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, None)



        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=self.fast)
            translations = builder1.from_batch(batch_data,tag = "src_vocabs1")

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                if len(n_best_preds[0]) == 0:
                    n_best_preds[0] = "unknown"
                self.out_file1.write(n_best_preds[0] + '\n')
                self.out_file1.flush()


        return all_scores, all_predictions

    def translate_batch(self, batch, data, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._translate_batch(batch, data)

    def _run_encoder(self, batch, data_type):
        src = inputters.make_features(batch, 'src', data_type)
        src1 = inputters.make_features(batch, 'src1', data_type)
        src_lengths = None
        src_lengths1 = None
        if data_type == 'text':
            _, src_lengths = batch.src
            _, src_lengths1 = batch.src1
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        enc_states1, memory_bank1, src_lengths1 = self.model.encoder1(
            src1, src_lengths1)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank) \
                .long() \
                .fill_(memory_bank.size(0))

        if src_lengths1 is None:
            assert not isinstance(memory_bank1, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths1 = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank1) \
                .long() \
                .fill_(memory_bank1.size(0))
        return src, enc_states, memory_bank, src_lengths, src1, enc_states1, memory_bank1, src_lengths1

    def _decode_and_generate(self, decoder_input, memory_bank, batch, data,
                             memory_lengths, src_map=None,
                             step=None, batch_offset=None):

        if self.copy_attn:
            # Turn any copied words to UNKs (index 0).
            decoder_input = decoder_input.masked_fill(
                decoder_input.gt(len(self.fields["tgt"].vocab) - 1), 0)

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder2(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=step)

        # Generator forward.
        if not self.copy_attn:
            attn = dec_attn["std"]
            log_probs = self.model.generator2(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator2(dec_out.view(-1, dec_out.size(2)),
                                           attn.view(-1, attn.size(2)),
                                           src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = data.collapse_copy_scores(
                scores,
                batch,
                self.fields["tgt"].vocab,
                data.src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset)
            scores = scores.view(decoder_input.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn

    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths, src1, enc_states1, memory_bank1, src_lengths1 = self._run_encoder(
            batch, data_type)
        self.model.decoder.init_state(src, memory_bank, enc_states)
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]
        #print(tgt_in.size())
        dec_out, attns = self.model.decoder(tgt_in, memory_bank,
                                      memory_lengths=src_lengths)

        inin_state_for_decoder_1 = [dec_out[i] for i in batch.tgt2_index]
        self.model.decoder2.init_state(src1, memory_bank1, inin_state_for_decoder_1)
        # dec_out, attns = self.decoder_1(batch.tgt1, memory_bank1,
        #                                    memory_lengths=src_lengths1)

        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []
        results["batch"] = batch
        if "tgt1" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank1, src_lengths1, data, batch.src1_map
                if data_type == 'text' and self.copy_attn else None)
            self.model.decoder2.init_state(
                src1, memory_bank1, inin_state_for_decoder_1, with_cache=True)
        else:
            results["gold_score"] = [0] * batch_size

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map1 = (tile(batch.src1_map, beam_size, dim=1)
                    if data.data_type == 'text' and self.copy_attn else None)
        self.model.decoder2.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank1, tuple):
            memory_bank1 = tuple(tile(x, beam_size, dim=1) for x in memory_bank1)
        else:
            memory_bank1 = tile(memory_bank1, beam_size, dim=1)
        memory_lengths1 = tile(src_lengths1, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(10):
            if all((b.done() for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.get_current_state() for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = \
                self._decode_and_generate(inp, memory_bank1, batch, data,
                                          memory_lengths=memory_lengths1,
                                          src_map=src_map1, step=i)

            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths1[j]])
                select_indices_array.append(
                    b.get_current_origin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder2.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths, data, src_map):
        tgt_in = inputters.make_features(batch, 'tgt2')[:-1]

        log_probs, attn = \
            self._decode_and_generate(tgt_in, memory_bank, batch, data,
                                      memory_lengths=src_lengths,
                                      src_map=src_map)
        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]

        log_probs[:, :, tgt_pad] = 0
        gold = batch.tgt2[1:].unsqueeze(2)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores
