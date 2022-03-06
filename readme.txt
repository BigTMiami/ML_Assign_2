Anthony Menninger
amenninger3
CS 7641 Machine Learning
Assignment 2

Infrastructure Setup
* Uses Python 3.8 (May work with 3.6 or 3.7) 
* CODE: https://github.com/BigTMiami/ML_Assign_2
    * To download, run: git clone https://github.com/BigTMiami/ML_Assign_2.git
    * Two changes were made to the mlrose_hiive project and are found in the mlrose_changes folder. To use these changes, these files should be copied into the appropriate library site packages.  The code can be run without doing this.
* DATA: For the neural network section, the MNIST data is already loaded in the data folder.  
    * MNIST data: http://yann.lecun.com/exdb/mnist/ - Already loaded in the project data folder.
* Python Setup: There is a requirements.txt in the root project folder.  This will load all the needed python libraries.
    * Best practice is to use a virtual environment, such as virtualenv, to load libraries.
    * From the command line in the root project directory, install with command: pip install -r requirements.txt

Running the code
* Section 1
    * The code can be run from the command line in the project root using an argument system.  For each of the three supported problems (four_peaks, k_color, knapsack) one of the four supported algorithms should be called (rhc, sa, ga, mimic). A problem length must be included. Below is the basic format:
        python run_opt.py four_peaks sa 20
    * There are optional arguments that can be set for all algorithms:
        * -max_iterations 20 
        * -max_attempts 10
        * -seed 1  (Probably don't define this which will default to reproducing the paper results)
    * For each algorithm, there are some required values particular to it.  A set of values can be provided to be combined on the output charts:
        * rhc
            * -restarts 20 30 40
        * sa
            * -temperatures 1 2
            * -decays geom arith exp  (Only supported types)
        * ga    
            * -populations 100 200
            * -mutations 0.1 0.2
        * mimic
            * -populations 100 200
            * -keep percentages 0.1 0.2
    * Here are four complete strings to demonstrate for each problem.  The problem lengths are set low for demo purposes. 
        * python run_opt.py four_peaks rhc 20 -restarts 10 -max_iterations 20 -max_attempts 10 
        * python run_opt.py knapsack sa 20 -max_iterations 20 -max_attempts 10 -temperatures 1 10 100 -decays geom
        * python run_opt.py knapsack ga 20 -max_iterations 20 -max_attempts 10 -populations 100 200 -mutations 0.1
        * python run_opt.py k_color mimic 20 -max_iterations 20 -max_attempts 10 -populations 50 100 -keep_percents 0.1
    * A chart is produced on each run and placed in Document/Figures/working folder.  
    * The data is saved in experiments/{problem_name}/{problem_length} folder.
* Section 2
    * This uses the same model as in the first section with slightly different settings. The basic form is below.  The four supported optimization types are backprop, rhc, sa, ga.
        * python run_neural.py sa 
    * The one key new value i:
        * -epochs 10  (governs how many times the data will be trained.)
    * Only one value is allowed for the algorithm specific settings. (-restart 10  instead of --restarts 10 20 30).
    * Here are four complete strings for each optimizer type (backprop, rhc, sa, ga)
        * python run_neural.py backprop -epochs 10  
        * python run_neural.py rhc -epochs 10 -max_iters 20 -max_attempts 5 -restart 1
        * python run_neural.py sa -epochs 10 -max_iters 20 -max_attempts 5 -temperature 0.001 -min_temp 0.0001 -decay geom
        * !THIS TAKES A LONG TIME!
            * python run_neural.py ga -epochs 1 -max_iters 3 -max_attempts 2 -population 20 -mutation 0.01
    * Two charts are produced for each run - an error chart and a loss chart, and placed in Document/Figures/neural folder
    * In the same folder, a pickled run file is also placed.  The load_run_info(filename) function in the charting.py module can be used to load these files.
