from mpi4py import MPI
import numpy as np
import time
import logging
from inspect import signature
from firedrake import *

class MLMC_Solver:

    def __init__ (self, problem, levels, repetitions, comm=None, comm_limits=None):
        """
        arg: 
            problem (class) - MLMC_Solver object containing problem.
            levels (int) - number of levels in MLMC
            repetitions (list) - repetitions at each level starting at coarsest
        """
        self.check_inputs(levels, repetitions, comm, comm_limits)
        self.MLMCproblem = problem
        self.levels = levels
        self.repetitions = repetitions
        self._new_reps = repetitions

        self._result = None
        
        # Parallelisation related attributes
        self._comms = None
        self._did_calculate = None
        self._initial_comm = comm
        self._comm_ids = None
        
        # Initialise logger for print-outs
        self._logger = self.initialiseLogger()

        # Initialise the comms and send them to the problem
        if comm is not None:
            self._comms = []
            self._new_reps = []
            self._did_calculate = []
            self._comm_ids = []
            self.initialiseCommunicators(comm_limits)
            self.MLMCproblem.set_comms(self._comms, self._did_calculate, self._initial_comm)

    def solve(self):
        self.MLMCproblem.initialiseLevelList(self.levels)

        # Iterate through each level in hierarchy
        for i in range(self.levels):
            # Only rank 0 core prints when in parallel
            if self._comms is None or self._comms[i].Get_rank() == 0:
                self._logger.info("LEVEL {} - {} Samples".format(i+1, self.repetitions[i]))
                if self._comms is not None:
                    self._logger.info("COMM: {}/{} \t SIZE: {} \t {} Sample(s)".format(
                        self._comm_ids[i][1]+1, self._comm_ids[i][0], 
                        self._comms[i].Get_size(), self._new_reps[i]))

            self.MLMCproblem.newLevel(i) # Create P_level obj in soln list
            
            # Sampling now begins
            for j in range(self._new_reps[i]):
                if self._comms is None or self._comms[i].Get_rank() == 0:
                    self._logger.info("Sample {} of {}".format(j+1, self._new_reps[i]))
                
                self.MLMCproblem.addTerm(i) # Calculate result from sample

            # This corresponds to the inner sum in the MLMC eqn.
            self.MLMCproblem.averageLevel(i)
        
        # Outer sum in MLMC eqn.
        self._result, lvls = self.MLMCproblem.sumAllLevels()
        
        return self._result, lvls
    
    def initialiseCommunicators(self, comm_limits):
        # colour list is a list of 0's and 1's that gives a binary number
        colour_list = [0]
        new_comm = self._initial_comm
        # Decide how many communicator splits needed between each level
        splits = self.find_splits(new_comm.Get_size(), comm_limits)
        
        # iterate from level L to level 0
        for i in range(self.levels):
            # Carry out splitting
            for j in range(splits[-(i+1)]):
                colour = new_comm.Get_rank() % 2
                colour_list.append(colour)
                new_comm = new_comm.Split(color=colour)

            self._comms.insert(0, new_comm)
            
            num_comms = 2**(len(colour_list)-1)

            # Find decimal number of colour_num binary list
            colour_num = int("".join(map(str, colour_list)),2)
            
            # Make correction for level 0
            if num_comms > self._initial_comm.Get_size():
                num_comms = self._initial_comm.Get_size()
                colour_num = self._initial_comm.Get_rank()
            
            rep = self.repetitions[-(i+1)]//num_comms
            rep_r = self.repetitions[-(i+1)]%num_comms
            self._comm_ids.insert(0, [num_comms, colour_num])
            
            # Distribute remainder samples across cores
            if colour_num < rep_r:
                rep += 1
            self._new_reps.insert(0, int(rep))
            
            # If no reps performed ensure the correct average is taken at the end
            if rep == 0:
                self._did_calculate.insert(0, 0)
            else:
                self._did_calculate.insert(0, 1)
    
    def find_splits(self, size, limits):
        splits = []
        for i in range(self.levels):
            counter = 0
            while size > limits[-(i+1)]:
                size = size//2
                counter += 1
            splits.insert(0, counter)

        return splits

    def check_inputs(self, levels, repetitions, comm, comm_limits):
        assert isinstance(repetitions, list) and isinstance(levels, int), \
        ("Repetitions must be a list and levels an integer")
        
        assert len(repetitions) == levels, \
        ("The levels arguement is not equal to the number of entries in repetitions")
        
        if comm is not None:
            assert isinstance(comm, MPI.Intracomm) and isinstance(comm_limits, list), \
            ("For parallel execution comm and comm_limits inputs must be Intracomm and list types")
            
            assert len(comm_limits) == levels, \
            ("A limit must be set for each MLMC level")
            
            assert all(comm_limits[i]<=comm_limits[i+1] for i in range(len(comm_limits)-1)), \
            ("Communicator maximum size must decrease or stay the same with decreasing level")


    def initialiseLogger(self):
        logger = logging.logging.getLogger("MLMC-logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.logging.StreamHandler()
        logger.addHandler(file_handler)
        return logger
        

class MLMC_Problem:
    def __init__(self, problem_class, sampler, lvl_maker):
        """
        arg: 
            problem_class (class) - class initialised with one argument - one of
            level objects returned by lvl_maker() function. 
            It must also have a solve() method which takes one argument - 
            a sample returned by sampler() function - and returns  the solution 
            to the problem using that sample.

            sampler (func) - two argument function which takes the two
            level obects returned by the lvl_maker() function. Function returns 
            a tuple of two samples for each of the input level.
            If second input arg is None then return None for second item in output
            list.
            
            lvl_maker (func) - two argument function which takes two integer
            levels (counting from 0) and returns a tuple of two level objects at
            the specified levels. 
            If the second argument is < 0 then the returned level obect for that
            level is None.
        """
        self.check_inputs(problem_class, sampler, lvl_maker)
        self.problem_class = problem_class
        self.sampler = sampler
        self.lvl_maker = lvl_maker

        self._comms = None
        self._level_list = None
        self._result = None
        self._did_calculate = None
        self._initial_comm = None

    def check_inputs(self, problem_class, sampler, lvl_maker):
        assert callable(sampler) and callable(lvl_maker), \
        ("The sampler and level_maker inputs should be functions")
        
        assert hasattr(problem_class, "solve") and callable(problem_class.solve), \
        ("The input probem class needs a solve() method - see MLMC_Problem docstring")
        
        samp_param = signature(sampler).parameters
        lvl_param = signature(lvl_maker).parameters
        lvl_last_arg_default = list(lvl_param.values())[-1].default
        prob_solve_param = signature(problem_class.solve).parameters
        
        assert len(samp_param) == 2, \
        ("The sampler input should be a function which takes two inputs")
        
        assert len(lvl_param) == 3 and lvl_last_arg_default == MPI.COMM_WORLD, \
        ("The level maker input should be a function which takes 3 inputs, the third of \n"
        "which defaults to MPI.COMM_WORLD")
        
        assert len(prob_solve_param) == 2, \
        ("The solve method of the input problem class should take one argument")

    def set_comms(self, comms, did_calc, init_comm):
        self._comms = comms
        self._did_calculate = did_calc
        self._initial_comm = init_comm

    # List with entry for each level
    # When entry == False level calculation has not begun
    # When type(entry) == P_level obj calculation in progress (summing terms)
    # When type(entry) == float obj calculation on that level completed 
    def initialiseLevelList(self, levels):
        self._level_list = [False for i in range(levels)]
        self._result = np.array([0 for i in range(levels)], dtype=np.float64)

    def newLevel(self, level):
        # if second term is negative return None in 2nd output
        if self._comms != None:
            lvl_f, lvl_c = self.lvl_maker(level, level-1, self._comms[level])
        else:
            lvl_f, lvl_c = self.lvl_maker(level, level-1)

        self._level_list[level] = P_level(self.problem_class, self.sampler, lvl_f, lvl_c)

    def addTerm(self, level):
        self._level_list[level].calculate_term()
    
    def averageLevel(self, level): 
        avg = self._level_list[level].get_average()
        if avg is None:
            assert self._did_calculate[level] == 0,("No calculations done on "
            "communicator unexpectedly")
            self._level_list[level] = 0.0
        else:
            self._level_list[level] = avg

    def sumAllLevels(self):
        assert all(isinstance(x, float) for x in self._level_list)
        
        if self._comms is not None :
            self._initial_comm.Reduce([np.array(self._level_list, dtype=np.float64), MPI.DOUBLE], 
            [self._result, MPI.DOUBLE], op=MPI.SUM, root=0)
            
            calculations_made = np.ones_like(self._did_calculate, dtype=np.float64)
            self._initial_comm.Reduce([np.array(self._did_calculate, dtype=np.float64), MPI.DOUBLE], 
            [calculations_made, MPI.DOUBLE], op=MPI.SUM, root=0)

            logging.warning(calculations_made)
            self._level_list = self._result/ calculations_made
        
        self._result = sum(self._level_list)
        
        return self._result, self._level_list

# HELPER CLASS
class P_level:
    def __init__(self, problem_class, sampler, lvl_f, lvl_c):
        # Assignments
        self._lvl_f = lvl_f
        self._lvl_c = lvl_c

        self.sampler = sampler
        self._value = None
        self._sample_counter = 0

        self.problem_f = problem_class(self._lvl_f)
        if self._lvl_c is not None:
            self.problem_c = problem_class(self._lvl_c)

        self.prob_class = problem_class

    def get_average(self):
        if self._value is None:
            return None
        return self._value/self._sample_counter
        

    def calculate_term(self):
        """
        Calculates result from new sample and adds it to _value. This is 
        equivalent to inner sum in MLMC equation.
        """
        if self._value is None:
            self._value = 0

        # Generate sample and solve problems with sample
        sample_f, sample_c = self.sampler(self._lvl_f, self._lvl_c)
        print(sample_f)
        print(self.problem_f._V.mesh().num_faces())
        e_f = self.problem_f.solve(sample_f)

        if self._lvl_c is not None:  
            e_c = self.problem_c.solve(sample_c)
            level1 = self.prob_class(FunctionSpace(UnitSquareMesh(40,40), "CG", 2))
            level0 = self.prob_class(FunctionSpace(UnitSquareMesh(20,20), "CG", 2))
            print(sample_f, sample_c)
            print(level1.solve(sample_f), level0.solve(sample_c))
            print(e_f, e_c)
            self._value +=  e_f - e_c
        else:
            print("**")
            print(e_f)
            self._value += e_f
        
        self._sample_counter += 1
        return 0


def do_MC(problem_class, repititions, level_ob, sampler):
    solutions = []
    s = time.time()
    prob = problem_class(level_ob)
    for i in range(repititions):
        print("Sample {} of {}".format(i+1, repititions))
        new_sample, x = sampler(level_ob, None)

        solutions.append(prob.solve(new_sample))
    print("Total time: {}".format(time.time()-s))
    return solutions


    
