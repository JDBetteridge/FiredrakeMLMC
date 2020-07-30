import time
import logging

class MLMC_Solver:

    def __init__ (self, problem, levels, repetitions):
        """
        arg: 
            problem (class) - MLMC_Solver object containing problem.
            levels (int) - number of levels in MLMC
            repetitions (list) - repetitions at each level starting at coarsest
        """
        self.MLMCproblem = problem
        self.levels = levels
        self.repetitions = repetitions

        self._result = None

    def solve(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        start = time.time()

        assert len(self.repetitions) == self.levels, \
        ("The levels arguement is not equal to the number of entries in repetitions")

        self.MLMCproblem.initialise_level_list(self.levels)

        # Iterate through each level in hierarchy
        for i in range(self.levels):
            logging.info("LEVEL {} - {} Samples".format(i+1, self.repetitions[i]))

            self.MLMCproblem.newLevel(i) # Create P_level obj in soln list
            
            # Sampling now begins
            for j in range(self.repetitions[i]):
                logging.info("Sample {} of {}".format(j+1, self.repetitions[i]))
                s2 = time.time()
                self.MLMCproblem.addTerm(i) # Calculate result from sample
                print("tot: {}".format(time.time()-s2))

            # This corresponds to the inner sum in the MLMC eqn.
            self.MLMCproblem.averageLevel(i)
        
        # Outer sum in MLMC eqn.
        self._result = self.MLMCproblem.sumAllLevels()
        
        end = time.time()
        logging.info("Runtime: {}s".format(end - start))

        return self._result
    

class MLMC_Problem:
    def __init__(self, problem_class, sampler, lvl_maker):
        """
        arg: 
            problem_class (class) - class initialised with one argument - one of
            level objects returned by lvl_maker() function. 
            It must also have a .solve() method which takes one argument - 
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
            If the second argument is None then the returned level obect for that
            level is None too.
        """
        self.problem_class = problem_class
        self.sampler = sampler
        self.lvl_maker = lvl_maker

        self._level_list = None
        self._result = None

        # List with entry for each level
        # When entry == False level calculation has not begun
        # When type(entry) == P_level obj calculation in progress (summing terms)
        # When type(entry) == float obj calculation on that level completed
        
      
    def initialise_level_list(self, levels):
        self._level_list = [False for i in range(levels)]
        self._result = None

    def newLevel(self, level):
        # add communicator in here
        # if second term is negative return None in 2nd output
        lvl_f, lvl_c = self.lvl_maker(level, level-1)
        self._level_list[level] = P_level(self.problem_class, self.sampler, lvl_f, lvl_c)

    def addTerm(self, level):
        self._level_list[level].calculate_term()
    
    def averageLevel(self, level):
        self._level_list[level] = self._level_list[level].get_average()

    def sumAllLevels(self):
        assert all(isinstance(x, float) for x in self._level_list)
        self._result = sum(self._level_list)
        return self._result

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
        if self._lvl_c != None:
            self.problem_c = problem_class(self._lvl_c)


    def get_average(self):
        self._hierarchy = None # For memory conservation clear hierarchy
        return self._value/self._sample_counter
        

    def calculate_term(self):
        """
        Calculates result from new sample and adds it to _value. This is 
        equivalent to inner sum in MLMC equation.
        """
        if self._value == None:
            self._value = 0

        # Generate sample and solve problems with sample
        sample_f, sample_c = self.sampler(self._lvl_f, self._lvl_c)
        e_f = self.problem_f.solve(sample_f)

        if self._lvl_c != None:  
            e_c = self.problem_c.solve(sample_c)
            self._value +=  e_f - e_c
        else:
            self._value += e_f
        
        self._sample_counter += 1
        return 0




    