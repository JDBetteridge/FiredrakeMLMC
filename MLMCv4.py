import matplotlib.pyplot as plt
import time
import json
import logging



def MLMC_general_scalar(problem_class, sampler, levels, repetitions, isEval=True):
    """
    arg: problem_class (func) - class initialised with one argument - the level
         as an integer. It must also have a .solve() method which takes one 
         argument - a sample - and returns  the solution to the problem using
         that sample
         sampler (func) - single argument function in which the integer level
         is entered and a tuple is returned giving a sample in the correct form 
         for both the coarse and fine level (sample_coarse, sample_fine)
         levels (int) - number of levels in MLMC
         repetitions (list) - repetitions at each level starting at coarsest
         isEval (bool) - whether or not evaluation should be run on result
    output: Estimate of value
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    start = time.time()

    assert len(repetitions) == levels, \
    ("The levels arguement is not equal to the number of entries in repetitions")

    solver = MLMC_Solver(problem_class, sampler, levels)

    # Iterate through each level in hierarchy
    for i in range(levels):
        logging.info("LEVEL {} - {} Samples".format(i+1, repetitions[i]))

        solver.newLevel(i) # Create P_level obj in soln list
        
        # Sampling now begins
        for j in range(repetitions[i]):
            logging.info("Sample {} of {}".format(j+1, repetitions[i]))

            solver.addTerm(i) # Calculate result from sample

        # This corresponds to the inner sum in the MLMC eqn.
        solver.averageLevel(i)
    
    # Outer sum in MLMC eqn.
    estimate = solver.sumAllLevels()
    
    end = time.time()
    logging.info("Runtime: ", end - start, "s")

    if isEval:
        solver.eval_result()

    return estimate



class MLMC_Solver:
    def __init__(self, problem_class, sampler, levels):
        self.problem_class = problem_class
        self.sampler = sampler
        self._levels = levels

        # List with entry for each level
        # When entry == False level calculation has not begun
        # When type(entry) == P_level obj calculation in progress (summing terms)
        # When type(entry) == float obj calculation on that level completed
        self._level_list = [False for i in range(levels)]

        self._result = None

    def newLevel(self, level):
        self._level_list[level] = P_level(self.problem_class, self.sampler, level)

    def addTerm(self, level):
        self._level_list[level].calculate_term()
    
    def averageLevel(self, level):
        self._level_list[level] = self._level_list[level].get_average()

    def sumAllLevels(self):
        assert all(isinstance(x, float) for x in self._level_list)
        self._result = sum(self._level_list)
        return self._result
    
    def eval_result(self):
        
        with open("10_int.json") as handle:
            e_10 = json.load(handle)
        
        with open("100_int.json") as handle:
            e_100 = json.load(handle)
        
        with open("1000_int.json") as handle:
            e_1000 = json.load(handle)
        
        with open("10000_int.json") as handle:
            e_10000 = json.load(handle)
        
        with open("20000_int.json") as handle:
            e_20000 = json.load(handle)

        d_10 = self._result - e_10
        d_100 = self._result - e_100
        d_1000 = self._result - e_1000
        d_10000 = self._result - e_10000
        d_20000 = self._result - e_20000

        print("% difference from 10 sample MC: ",(d_10*100)/self._result)
        print("% difference from 100 sample MC: ",(d_100*100)/self._result)
        print("% difference from 1000 sample MC: ",(d_1000*100)/self._result)
        print("% difference from 10000 sample MC: ",(d_10000*100)/self._result)
        print("% difference from 20000 sample MC: ",(d_20000*100)/self._result)

        convergence_tests(self._result)



class P_level:
    
    def __init__(self, problem_class, sampler, level):
        # Assignments
        self._level = level
        self.sampler = sampler
        self._value = None
        self._sample_counter = 0

        self.problem_f = problem_class(self._level)
        if self._level >= 1:
            self.problem_c = problem_class(self._level-1)
            

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
        sample_c, sample_f = self.sampler(self._level)
        e_f = self.problem_f.solve(sample_f)

        if self._level >= 1:  
            e_c = self.problem_c.solve(sample_c)
            self._value +=  e_f - e_c
        else:
            self._value += e_f
        
        self._sample_counter += 1
        return 0


def convergence_tests(param = None):
    """
    Function which compares result to 10,000 sample MC 
    """
    with open("20000_list.json") as handle:
            results = json.load(handle)
    
    res2 = [sum(results[:i+1])/(i+1) for i in range(len(results))]
    #print(res2[0], results[0])
    fig, axes = plt.subplots()
    axes.plot([i for i in range(20000)], res2, 'r')
    if param != None:
        plt.axhline(y=param, color='b')
    #axes.hist(solutions, bins = 40, color = 'blue', edgecolor = 'black')
    plt.show()


    