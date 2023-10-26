.. currentmodule:: slideflow.io.torch

slideflow.io.torch
===================

The purpose of this module is to provide a performant, backend-agnostic TFRecord reader and interleaver to use as
input for PyTorch models. Its TFRecord reader is a modified and optimized version of
https://github.com/vahidk/tfrecord, included as the module :mod:`slideflow.tfrecord`. TFRecord file reading and
interleaving is supervised by :func:`slideflow.io.torch.interleave`, while the
:func:`slideflow.io.torch.interleave_dataloader` function provides a PyTorch DataLoader object which can be directly used.

.. automodule:: slideflow.io.torch
    :members:
    :exclude-members: StyleGAN2Interleaver, TileLabelInterleaver, InterleaveIterator, IndexedInterleaver

.. autoclass:: slideflow.io.torch.InterleaveIterator

.. autoclass:: slideflow.io.torch.IndexedInterleaver