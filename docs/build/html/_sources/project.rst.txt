Setting up a Project
====================

The easiest way to get the ``slideflow`` pipeline up and running is to use the bundled project management class, ``SlideFlowProject``. In this section, we will examine how to set up a new project and then use the project to execute each of the pipeline steps. 

Before we start, make sure you have each of the following:

1.	A collection of slides, preferably in SVS format though JPG images are also supported.
2.	A collection of ROIs in CSV format, generated using QuPath.
3.	A plan for which slides will be used for training and which will be used for final testing.

The script we will use to manage our project is ``run_project.py``. An empty version of this script that will setup a project but not perform any pipeline actions is included below:

To use the script to execute pipeline actions, simply run the script:

.. code-block:: bash

	$ python3 run_project.py

Project Configuration
*********************

Upon first executing the script, you will be asked a series of questions regarding your project. Default answers are given in brackets (if the question is a yes/no question, the default answer is the letter which is capitalized); if you press enter without typing anything, the default will be chosen. You can always change your answers later by editing ``settings.json`` in your project folder. Below is an overview of what you’ll be asked for.