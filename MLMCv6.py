from mpi4py import MPI
import numpy as np
import time
import logging

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
        self._comms = None
        self._did_calculate = None
        
        self._logger = self.initialise_logger()

        # Initialise the comms and send them to the problem
        if comm is not None:
            self._comms = []
            self._new_reps = []
            self._did_calculate = []
            self.initialise_communicators(comm, comm_limits)
            self.MLMCproblem.set_comms(self._comms, self._did_calculate)

    def solve(self):
        start = time.time()

        self.MLMCproblem.initialise_level_list(self.levels)

        # Iterate through each level in hierarchy
        for i in range(self.levels):
            self._logger.info("LEVEL {} - {} Samples".format(i+1, self.repetitions[i]))

            self.MLMCproblem.newLevel(i) # Create P_level obj in soln list
            
            # Sampling now begins
            for j in range(self._new_reps[i]):
                self._logger.info("Sample {} of {}".format(j+1, self._new_reps[i]))
                
                self.MLMCproblem.addTerm(i) # Calculate result from sample

            # This corresponds to the inner sum in the MLMC eqn.
            self.MLMCproblem.averageLevel(i)
        
        # Outer sum in MLMC eqn.
        self._result, lvls = self.MLMCproblem.sumAllLevels()
        
        end = time.time()
        self._logger.info("Runtime: {}s".format(end - start))

        return self._result, lvls
    
    def initialise_communicators(self, entered_comm, comm_limits):
        # colour list is a list of 0's and 1's that gives a binary number
        colour_list = [0]
        new_comm = entered_comm
        for i in range(self.levels):
            # decide whether to split
            while new_comm.Get_size() > comm_limits[-(i+1)][1]:
                colour = new_comm.Get_rank() % 2
                colour_list.append(colour)
                new_comm = new_comm.Split(color=colour)

            self._comms.insert(0, new_comm)
            num_comms = entered_comm.Get_size() / new_comm.Get_size()
            

            rep = self.repetitions[-(i+1)]//num_comms
            rep_r = self.repetitions[-(i+1)]%num_comms
            
            # Find decimal number of colour_num binary list
            colour_num = int("".join(map(str, colour_list)),2)
            
            # decimal number found should always be less than num_comms
            assert num_comms == 2**len(colour_list-1), \
            ("Communicator Division Error: Must be an even number of cores in COMM_WORLD")
            # Distribute remainder across cores
            if colour_num < rep_r:
                rep += 1
            self._new_reps.insert(0, int(rep))
            
            # If no reps performed make not to ensure the correct average is taken
            if rep == 0:
                self._did_calculate.insert(0, 0)
            else:
                self._did_calculate.insert(0, 1)

    def check_inputs(self, levels, repetitions, comm, comm_limits):
        assert len(repetitions) == levels, \
        ("The levels arguement is not equal to the number of entries in repetitions")
        if comm is not None:
            assert comm_limits is not None, \
            ("Must enter both comm and comm_limits arguments for parallel processing")
            assert all(i[0]*2 <= i[1] for i in comm_limits), \
            ("Comm max and min limits need to be different by at least a factor of two")
            assert comm.Get_size() >= comm_limits[-1][0], \
            ("Input communicator must be at least as large as min comm for highest level")


    def initialise_logger(self):
        logger = logging.getLogger("MLMC-logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.StreamHandler()
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
        assert hasattr(problem_class, "solve") and callable(problem_class.solve), \
        ("The input probem class needs a solve() method - see MLMC_Problem docstring")
        self.problem_class = problem_class
        self.sampler = sampler
        self.lvl_maker = lvl_maker

        self._comms = None
        self._level_list = None
        self._result = None
        self._did_calculate = None



    def set_comms(self, comms, did_calc):
        self._comms = comms
        self._did_calculate = did_calc

    # List with entry for each level
    # When entry == False level calculation has not begun
    # When type(entry) == P_level obj calculation in progress (summing terms)
    # When type(entry) == float obj calculation on that level completed 
    def initialise_level_list(self, levels):
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
            assert self._did_calculate[level] == 0,("No calculations done communicator unexpectedly")
            self._level_list[level] = 0.0
        else:
            self._level_list[level] = avg


        #print(self._level_list)

    def sumAllLevels(self):
        assert all(isinstance(x, float) for x in self._level_list)
        #print(self._level_list)
        if self._comms is not None :
            self._comms[-1].Reduce([np.array(self._level_list, dtype=np.float64), MPI.DOUBLE], 
            [self._result, MPI.DOUBLE], op=MPI.SUM, root=0)
            
            calculations_made = np.ones_like(self._did_calculate, dtype=np.float64)
            self._comms[-1].Reduce([np.array(self._did_calculate, dtype=np.float64), MPI.DOUBLE], 
            [calculations_made, MPI.DOUBLE], op=MPI.SUM, root=0)

            logging.warning(calculations_made)
            self._level_list = self._result/ calculations_made
            #self._result = sum(self._result/ calculations_made)
        
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
        e_f = self.problem_f.solve(sample_f)

        if self._lvl_c is not None:  
            e_c = self.problem_c.solve(sample_c)
            self._value +=  e_f - e_c
        else:
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


    
