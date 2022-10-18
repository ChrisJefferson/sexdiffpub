This is the code from the paper: "Agent-based models of the cultural evolution of occupational gender roles"

Note that re-generating all the experiments from this paper will take several days on a 64-core computer!

Requirements
------------

* Python 3.9 or above
* The program 'gnuparallel' (to run experiments in parallel)


Quick Instructions
------------------

The following commands should be cut + pasted into the command line. Lines starting '#' are comments, describing each command.

```
# Set up a python 'virtual environment' to install packages into
python -m venv env
# Enter the 'virtual environment'
. env/bin/activate
# Install required packages
pip install -r requirements.txt
# Run all the experiments
# Note, this will take several days
./make_graphs.sh | parallel
# Merge results into csv files
./merge_results.sh
```
