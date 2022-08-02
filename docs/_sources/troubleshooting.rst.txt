Troubleshooting
===============

If you're running into problems, look for more information by including debug logging. To enable debug logging, set the environmental variable ``SF_LOGGING_LEVEL=10``.

To check for errors in your environment or installation, you can also use the test script ``test.py``, which uses the testing suite ``slideflow.test.TestSuite`` to execute all pipeline functions on a set of provided slides.

Testing
*******

To troubleshoot environment or installation issues, start by running unit tests,
which do not require any sample slides. Use the ``test.py`` script without any
arguments:

.. code-block:: bash

    $ python3 test.py

For a more comprehensive test of all pipeline functions, provide a path to a directory containing sample slides via ``--slides``, setting ``--all=True`` to run all tests:

.. code-block:: bash

    $ python3 test.py --slides=/path/to/slides --all=True

Individual tests can be manually skipped with the following syntax:

.. code-block:: bash

    $ python3 test.py --slides=/path/to/slides --all=True --reader=False

To view a list of all tests that will be run (and thus can be skipped), pass the argument ``--help``.

Issue Reporting
***************

If the issue is still unclear, please submit an Issue on the `project Github page <https://github.com/jamesdolezal/slideflow/issues>`_.
