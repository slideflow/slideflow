.. currentmodule:: slideflow.io.tensorflow

slideflow.io.tensorflow
=======================

TFRecord interleaving in the Tensorflow backend is accomplished with :func:`slideflow.io.tensorflow.interleave`, which interleaves a set of tfrecords together into a :class:`tf.data.Datasets` object that can be used for training. This interleaving can include patient or category-level balancing for returned batches (see :ref:`balancing`).

.. note::
    The TFRecord reading and interleaving implemented in this module is only compatible with Tensorflow models. The :mod:`slideflow.io.torch` module includes a PyTorch-specific TFRecord reader.

.. automodule:: slideflow.io.tensorflow
    :members: