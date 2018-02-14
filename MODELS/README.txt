Files:

MODEL.py & MODEL_combined.py: Code to build the Hierarchical Multivariate Hawkes Model and its combined version and simulate timeseries. Before using it, the required parameters have to be trained using the files inside_sesson_multi(combined).py and sessions_distributions.py.

inside_session_multi.py & inside_session_multi_combined.py: Programs that train the parameters of the Multivariate Hawkes Process using the data from the folder "DATA".

sessions_distributions.py: computes the empirical daily and weekly session distribution and the session length distribution for each user and for the whole dataset.

data.py: class with with functions to load and prepare the data from "DATA" used in different.

plot.py: functions to make different plots. Some are used in other programs.

comparator.py: program used to compare two different Hierarchical Multivariat Hawkes Models by performing simulations with each model and comparing the average results.

measure_hawkes_error.py: for different number of training sessions, train the parameters several times and return the results.

multi_combined_result_analyser.py train the hawkes parameters for the dataset "dataset name" several times and save the execution with minimum negative log likelihood.



