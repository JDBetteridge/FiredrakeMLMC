
L_1, L_0 = level_maker(1, 0) # Create level objects
P_0 = Problem_Class(L_0) # Initialise problem on level 0
P_1 = Problem_Class(L_1) # Initialise problem on level 1

Y = 0 # Level 1 result
N1 = 10 # Samples on level 1

# Generate and solve on N1 random samples
for n in range(N1):
    sample_1, sample_0 = sampler(L_1, L0)
    Y += (P_1.solve(sample_1) - P_0.solve(sample_0))

Y /= N1