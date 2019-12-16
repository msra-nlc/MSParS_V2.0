""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder,encoder1, decoder, decoder_1, decoder_2):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder1 = encoder1
        self.decoder1 = decoder_1
        self.decoder2 = decoder_2

    def forward(self, src, src1, tgt, tgt1, tgt2, tgt1_index, tgt2_index, lengths, lengths1):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt1 = tgt1[:-1]  # exclude last target from inputs
        tgt2 = tgt2[:-1]  # exclude last target from inputs

        enc_state_1, memory_bank_1, memory_lengths_1 = self.encoder(src, lengths)
        enc_state_2, memory_back_2, memory_lengths_2 = self.encoder1(src1, lengths1)
        #print(enc_state_1)
        self.decoder.init_state(src, memory_bank_1, enc_state_1)
        dec_out, attns = self.decoder(tgt, memory_bank_1,
                                      memory_lengths=memory_lengths_1)

        inin_state_for_decoder_1 = [dec_out[i] for i in tgt1_index]
        init_state_for_decoder_2 = [dec_out[i] for i in tgt2_index]
        #print(inin_state_for_decoder_1)
        self.decoder1.init_state(src1, memory_back_2, inin_state_for_decoder_1)
        self.decoder2.init_state(src1, memory_back_2, init_state_for_decoder_2)

        dec_out_1, attns_1 = self.decoder1(tgt1, memory_back_2,
                                      memory_lengths=memory_lengths_2)

        dec_out_2, attns_2 = self.decoder2(tgt2, memory_back_2,
                                      memory_lengths=memory_lengths_2)



        return dec_out, attns, dec_out_1, attns_1, dec_out_2, attns_2
