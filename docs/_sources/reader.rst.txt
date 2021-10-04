.. currentmodule:: slideflow.io.torch

slideflow.io.torch
===================

The purpose of this module is to provide a performant, backend-agnostic TFRecord reader and interleaver to use as
input for Torch models. It uses a backend-agnostic TFRecord reader based on a modified version of
the tfrecord reader/writer https://github.com/vahidk/tfrecord, optimized with improved performance. Binary TFRecord file
reading and interleaving is supervised by :func:`slideflow.io.torch.interleave`, while the
:func:`slideflow.io.torch.interleave_dataloader` function provides a PyTorch DataLoader object which can be directly used.

.. warning::
    PyTorch support is currently in development. The interleaving functions in this module are currently optimized for
    throughput at the expense of memory usage; if memory usage becomes excessive with large datasets, consider decreasing
    `num_workers` and `chunk_size`.

.. automodule:: slideflow.io.torch
    :members: