This software package implements the algorithms described in the paper

> Roger B. Grosse, Ruslan Salakhutdinov, William T. Freeman, and Joshua B. Tenenbaum,
> "Exploiting compositionality to explore a large space of model structures," UAI 2012.


# Requirements

This code base depends on a number of Python packages, most of which are pretty standard.
Most of the packages are available through [Enthought Canopy](https://www.enthought.com/products/canopy/),
which all academic users (including professors and postdocs) can use for free under their
[academic license](https://www.enthought.com/products/canopy/academic/). We use the following
Python packages which are included in Canopy:

- [NumPy](http://www.numpy.org/) (I used 1.6.1)
- [Matplotlib](http://matplotlib.org/index.html) (I used 1.2.0)
- [SciPy](http://www.scipy.org/) (I used 0.12.0)
- [scikit-learn](http://scikit-learn.org/stable/)  (I used 0.13.1)

Note: I've been told that [Anaconda Python](https://store.continuum.io/cshop/anaconda/) is an
alternative distribution which includes these same packages, has a comparable academic license,
and is easier to get running. I've never tried it myself, though.

There are two additional requirements, which are both `easy_install`able:

- [termcolor](https://pypi.python.org/pypi/termcolor)
- [progressbar](https://code.google.com/p/python-progressbar/)

More recent versions than the ones listed above should work fine, though unfortunately
the interfaces to some SciPy routines have a tendency to change without warning...

Also, if you want to distribute jobs across multiple cores or machines (highly recommended), you
will need to do one of the following:

- install [GNU Parallel](www.gnu.org/software/parallel) (see Configuration section for more details)
- write a scheduler which better matches your own computing resources (section TODO)


# Configuration

In order to run the structure search, you need to specify some local configuration parameters
in `config.py`. First, in the main project directory, copy the template:

    cp config_example.py config.py

In `config.py`, you need to specify the following paths:

- `CODE_PATH`, the directory where you keep the code for this project
- `CACHE_PATH`, a directory for storing intermediate results (which can take up a fair amount of disk
  space and are OK to delete when the experiment is done) (TODO: this will probably be changed)
- `RESULTS_PATH`, the directory for storing the machine-readable results of the structure search
- `REPORT_PATH`, the directory for saving human-readable reports

You also need to specify `SCHEDULER` to determine how the experiment jobs are to be run. The 
choices are `'single_process'`, which runs everything in a single process (not practical except
for the smallest matrices), and `'parallel'`, which uses GNU Parallel to distribute the jobs
across different machines, or different processes on the same machine. If you use GNU Parallel,
you also need to specify:

- `JOBS_PATH`, a directory for saving the status of jobs, if you are using GNU Parallel (TODO:
  this will probably be changed)
- `DEFAULT_NUM_JOBS`, the number of jobs to run on each machine

Note that using our GNU Parallel wrapper requires the ability to `ssh` into the machines without
entering a password. We realize this might not correspond to your situation, so TODO describes
how you can write your own job scheduler module geared towards the clusters at your own institution.


# Running the example

We provide an example of how to run the structure search in `example.py`. This runs the
structure search on the mammals dataset of Kemp et al. (2006), "Learning systems of concepts
with an infinite relational model." This is a 50 x 85 matrix where the rows represent
different species of mammal, the columns represent attributes, and each entry is a binary
value representing subjects' judgments of whether the animal has that attribute. Our structure
search did not result in a clear structure for this dataset, but it serves as an example which
can be run quickly (2 CPU minutes for me). 

After following the configuration directions above, run the following from the command line:

    python example.py
    python experiments.py everything example

This will run the structure search, and then output the results to the shell (and also save
them to the `example` subdirectory of `config.REPORT_PATH`). The results include the following:

- the best-performing structure at each level of the search, with their improvement in
  predictive log-likelihood for rows and columns, as well as z-scores for the improvement
  (see TODO)
- the total CPU time, also broken down by model
- the predictive log-liklihood scores for all structures at all levels of the search, sorted 
  from best to worst

Note that the search parameters used in this example are probably
insufficient for inference; if you are interested in accurate results for this dataset,
change `QuickParams` to `SmallParams` in `example.py`.



# Running the structure search

Suppose you have a real-valued matrix `X` you're interested in learning the structure of,
in the form of a NumPy array. The first step is to create a `DataMatrix` instance:

    from observations import DataMatrix
    data_matrix = DataMatrix.from_real_values(X)

This constructor also takes some optional arguments: 

- `mask`, which is a binary array determining which entries of `X` are observed. (By default,
  all entries are assumed to be observed.)
- `row_label` and `col_label`, which are Python lists giving the label of each row or column.
  These are used for printing the learned clusters and binary components.

The code doesn't do any preprocessing of the data, so it's recommended that you standardize
it to have zero mean and unit variance.

Next, you want to initialize an experiment for this matrix. You do this by passing in the
`DataMatrix` instance, along with a parameters object. `experiments.SmallParams` gives a
reasonable set of defaults for small matrices (e.g. 200 x 200), and `experiments.LargeParams`
gives a reasonable set of defaults for larger matrices (e.g. 1000 x 1000). This creates a
subdirectories of `config.RESULTS_PATH` and `config.REPORT_PATH` where all the computations
and results will be stored. For example,

    from experiments import init_experiment, LargeParams
    init_experiment('experiment_name', data_matrix, LargeParams())

You can also override the default parameters by passing keyword arguments to the parameters
constructor. See `experiments.DefaultParams` for more details. Finally, from the command line,
run the whole structure search using the following:

    python experiments.py everything experiment_name
    
You can also specify some optional keyword arguments:

- `--machines`, the list of machines to distribute the jobs to if you are using GNU Parallel.
  This should be a comma-separated list with no spaces. By default, it runs jobs only on the same machine.
- `--njobs`, the number of jobs to run on each machine if you are using GNU Parallel. (This
  overrides the default value in `config.DEFAULT_NUM_JOBS`.)
- `--email`, your e-mail address, if you want it to e-mail you the report when it finishes.

For example,

    python experiments.py everything experiment_name --machines machine1,machine2,machine3 --njobs 2 --email me@example.com

If all goes well, a report will be saved to `experiment_name/results.txt` under `config.REPORT_PATH`.



