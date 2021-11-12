Troubleshooting
===============

If you're running into problems, look for more information by including debug logging. To enable debug logging, set the environmental variable ``SF_BACKEND=10``.

To check for errors in your environment or installation, you can also use the included testing suite ``slideflow.test.TestSuite`` to look for issues with executing pipeline functions, as described below.

Testing
*******

The ``slideflow.test.TestSuite`` can test all pipeline functions using a directory of available slides. To initialize the testing suite, use the following syntax:

.. code-block:: python

    from slideflow.test import TestSuite
    TS = TestSuite('/path/to/testing/directory', slides='/path/to/slides')

If ``slides`` is set to ``'download'``, a set of slides will be downloaded from The Cancer Genome Atlas (TCGA) for testing.

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