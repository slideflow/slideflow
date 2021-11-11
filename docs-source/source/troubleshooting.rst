Troubleshooting
===============

If you're running into problems, look for more information by including debug logging. To enable debug logging, use python's built-in ``logging`` class to set the ``'slideflow'`` logging level to ``logging.DEBUG`` prior to loading a model:

.. code-block:: python

    import logging
    logging.getLogger('slideflow').setLevel(logging.DEBUG)
    P = sf.Project(...)

To check for errors in your environment or installation, you can also use the included testing suite ``slideflow.test.TestSuite`` to look for issues with executing pipeline functions.

Testing
*******

The ``slideflow.test.TestSuite`` can use either a directory of already available slides, or it can download a set of example files from `The Cancer Genome Atlas <https://portal.gdc.cancer.gov/>`_ (TCGA) for testing. To initialize the testing suite, use the following syntax:

.. code-block:: python

    from slideflow.test import TestSuite
    TS = TestSuite('/path/to/testing/directory', slides='/path/to/slides')

If ``slides`` is not provided, a set of random slides will be downloaded from TCGA for testing.

For debugging and troubleshooting, you can use the verbosity argument to include debugging logs:

.. code-block:: python

    import logging
    TS = TestSuite(..., verbosity=logging.DEBUG)

To reset a test project, you can pass ``True`` to the argument ``reset``:

.. code-block:: python

    TS = TestSuite(..., reset=True)

Once the ``TestSuite`` object has been initialized, you can begin the series of test by using the ``test()`` function:

.. code-block:: python

    TS = TestSuite(...)
    TS.test()

Issue Reporting
***************

If the issue is still unclear, please submit an Issue on the `project Github page <https://github.com/pearson-laboratory/slideflow>`_.