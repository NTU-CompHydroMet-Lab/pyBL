.. _contribute_guideline:

Contributing to pybl
====================

Preqrequisites
--------------
- Python 3.10 or higher
- Poetry

Setup
-----
- **Install Poetry**

You can install poetry by following the instructions on their website: https://python-poetry.org/docs/#installation
We recommend using poetry to install all the dependencies, as it will also create a virtual environment for you.

- **Clone the repository**

Clone the repository to your local machine using git:

.. code-block:: bash

    git clone https://github.com/NTU-CompHydroMet-Lab/pyBL.git

- **Install dependencies**

Install the dependencies using poetry:

.. code-block:: bash

    poetry install

- **Test if everything works**

.. code-block:: bash

    tox

If everything works, you should see something like this:

.. code-block:: bash

    py39: OK (5.51=setup[2.39]+cmd[3.12] seconds)
    py310: OK (4.49=setup[1.60]+cmd[2.89] seconds)
    py311: OK (5.32=setup[1.91]+cmd[3.41] seconds)
    ruff: OK (1.54=setup[1.52]+cmd[0.03] seconds)
    mypy: OK (2.91=setup[1.54]+cmd[1.36] seconds)
    docs: OK (4.20=setup[1.57]+cmd[2.63] seconds)
    congratulations :) (24.02 seconds)
