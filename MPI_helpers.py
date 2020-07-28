import math

def process_division(size, levels):
    processors = [i for i in range(size)]
    assignment = [[] for i in range(levels)]
    current_level = levels-1

    while len(processors) > 1:
        reamaining = len(processors)
        if current_level != 0:
            for i in range(math.ceil(remaining/2)):
                assignment[current_level].append(processors.pop())
        else:
            assignment[current_level] = processors
        
        current_level -= 1
    
    if current_level != -1:
