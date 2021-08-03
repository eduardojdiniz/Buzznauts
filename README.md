# Buzznauts
NMA-DL 2021 - Buzznauts

## Organization of the  project

The project has the following structure:

    Buzznauts/
      |- README.md
      |- Buzznauts/
         |- __init__.py
         |- Buzznauts.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- setup.py
      |- .mailmap
      |- LICENSE
      |- Makefile
      |- scripts/
         |- ...


### Module code

We place the module code in a file called `Buzznauts.py` in directory called
`Buzznauts`. This structure is a bit confusing at first, but it is a simple way
to create a structure where when we type `import Buzznauts as buzz` in an
interactive Python session, the classes and functions defined inside of the
`Buzznauts.py` file are available in the `buzz` namespace. For this to work, we
need to also create a file in `__init__.py` which contains code that imports
everything in that file into the namespace of the project:

    from .Buzznauts import *

In the module code, we follow the convention that all functions are either
imported from other places, or are defined in lines that precede the lines that
use that function. This helps readability of the code, because you know that if
you see some name, the definition of that name will appear earlier in the file,
either as a function/variable definition, or as an import from some other module
or package.

In the case of the Buzznauts module, the main classes defined at the bottom of
the file make use of some of the functions defined in preceding lines.

Remember that code will be probably be read more times than it will be written.
Make it easy to read (for others, but also for yourself when you come back to
it), by following a consistent formatting style. We strongly recommend
following the
[PEP8 code formatting standard](https://www.python.org/dev/peps/pep-0008/), and
we enforce this by running a code-linter called
[`flake8`](http://flake8.pycqa.org/en/latest/), which automatically checks the
code and reports any violations of the PEP8 standard (and checks for other
  general code hygiene issues), see below.

### Project Data

Used to store (small) project data alongside the module code.  Even if the data
that we are analyzing is too large, and cannot be effectively tracked with
github, we might still want to store some data for testing purposes.

Either way, we created a `Buzznauts/data` folder in which you can organize the
data. In the test scripts, and in the analysis scripts, the following snippet
provides a standard way to load the file-system location for the data:

    import os.path as op
    import Buzznauts as buzz
    data_path = op.join(buzz.__path__[0], 'data')


### Testing

We are using the ['pytest'](http://pytest.org/latest/) library for
testing. The `py.test` application traverses the directory tree in which it is
issued, looking for files with the names that match the pattern `test_*.py`
(typically, something like our `Buzznauts/tests/test_Buzznauts.py`). Within each
of these files, it looks for functions with names that match the pattern
`test_*`. Typically each function in the module would have a corresponding test .
This is sometimes called 'unit testing', because it independently tests each atomic
unit in the software. Other tests might run a more elaborate sequence of functions
('end-to-end testing' if you run through the entire analysis), and check that
particular values in the code evaluate to the same values over time. This is
sometimes called 'regression testing'. Regressions in the code are often buzzwards
in the Deep Learning zoo, telling you that you need to examine changes in your software
dependencies, the platform on which you are running your software, etc.

Test functions should contain assertion statements that check certain relations
in the code. Most typically, they will test for equality between an explicit
calculation of some kind and a return of some function.

To run the tests on the command line, change your present working directory to
the top-level directory of the repository (e.g. `/home/eduardojdiniz/proj/Buzznauts`),
and type:

    py.test Buzznauts

This will exercise all of the tests in your code directory. If a test fails, you
will see a message such as:


    Buzznauts/tests/test_Buzznauts.py .F...

    =================================== FAILURES ===================================
    ...

    Buzznauts/tests/test_Buzznauts.py:49: AssertionError
    ====================== 1 failed, 4 passed in 0.82 seconds ======================

This indicates to you that a test has failed.

As your code grows and becomes more complicated, you might develop new features
that interact with your old features in all kinds of unexpected and surprising
ways. As you develop new features of your code, keep running the tests, to make
sure that you haven't broken the old features.  Keep writing new tests for your
new code, and recording these tests in your testing scripts. That way, you can
be confident that even as the software grows, it still keeps doing correctly at
least the few things that are codified in the tests.

We have also provided a `Makefile` that allows you to run the tests with more
verbose and informative output from the top-level directory, by issuing the
following from the command line:

    make test

### Styling

It is a good idea to follow the PEP8 standard for code formatting. Common code
formatting makes code more readable, and using tools such as `flake8` (which
combines the tools `pep8` and `pyflakes`) can help make your code more readable,
avoid extraneous imports and lines of code, and overall keep a clean project
code-base.

Some projects include `flake8` inside their automated tests, so that every pull
request is examined for code cleanliness.

In this project, we have run `flake8` most (but not all) files, on
most (but not all) checks:

```
flake8 --ignore N802,N806 `find . -name *.py | grep -v setup.py | grep -v /doc/`
```

This means, check all .py files, but exclude setup.py and everything in
directories named "doc". Do all checks except N802 and N806, which enforce
lowercase-only names for variables and functions.

The `Makefile` contains an instruction for running this command as well:

    make flake8

### Documentation

TODO


### Installation

TODO

For installation and distribution we will use the python standard
library `setuptools` module. This module uses a `setup.py` file to
figure out how to install your software on a particular system. For a
small project such as this one, managing installation of the software
modules and the data is rather simple.

A `Buzznauts/version.py` contains all of the information needed for the
installation and for setting up the [PyPI
page](https://pypi.python.org/pypi/Buzznauts) for the software.
This also makes it possible to install your software with using `pip` and
`easy_install`, which are package managers for Python software. The
`setup.py` file reads this information from there and passes it to the
`setup` function which takes care of the rest.

Much more information on packaging Python software can be found in the
[Hitchhiker's guide to
packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.org).


### Continuous integration

TODO

### Distribution

TODO

### Licensing

License our code! A repository like this without a license maintains
copyright to the author, but does not provide others with any
conditions under which they can use the software. In this case, we use
the MIT license. You can read the conditions of the license in the
`LICENSE` file. As you can see, this is not an Apple software license
agreement (has anyone ever actually tried to read one of those?). It's
actually all quite simple, and boils down to "You can do whatever you
want with my software, but I take no responsibility for what you do
with my software"

For more details on what you need to think about when considering
choosing a license, see this
[article](http://www.astrobetter.com/blog/2014/03/10/the-whys-and-hows-of-licensing-scientific-code/)!

### Getting cited

When others use your code in their research, they should probably cite you. To
make their life easier, we use [duecredit](http://www.duecredit.org). This is a software
library that allows you to annotate your code with the correct way to cite it.
To enable `duecredit`, we have added a file `due.py` into the main directory.
This file does not need to change at all (though you might want to occasionally
update it from duecredit itself. It's
[here](https://github.com/duecredit/duecredit/blob/master/duecredit/stub.py),
under the name `stub.py`).

In addition, we will want to provide a digital object identifier (DOI) to the
article we want people to cite.

TODO

### Scripts

A scripts directory can be used as a place to experiment with your
module code, and as a place to produce scripts that contain a
narrative structure, demonstrating the use of the code, or producing
scientific results from your code and your data and telling a story
with these elements.


### Git Configuration

Currently there are two files in the repository which help working
with this repository, and which you could extend further:

- `.gitignore` -- specifies intentionally untracked files (such as
  compiled `*.pyc` files), which should not typically be committed to
  git (see `man gitignore`)
- `.mailmap` -- if any of the contributors used multiple names/email
  addresses or his git commit identity is just an alias, you could
  specify the ultimate name/email(s) for each contributor, so such
  commands as `git shortlog -sn` could take them into account (see
  `git shortlog --help`)

## TODO
Fix `REQUIRES` field on `./Buzznauts/version.py`
