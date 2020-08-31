from mpi4py import MPI
import logging
import math

#from randomgen import RandomGenerator, PCG64
from firedrake import *
from matern import matern
import pdb
#rg = RandomGenerator(PCG64(12345))
"""

world = MPI.COMM_WORLD
wrank = world.Get_rank()
wsize = world.Get_size()

qs = [2, 3, 4, 5]
#colour = world.Get_rank()%len(qs)
#comm = MPI.Comm.Split(world, colour)

ans = math.sqrt(qs[wrank])
ans = world.gather(ans, root=0)
print(ans)
#rank = comm.Get_rank()
#size = comm.Get_size()

logging.info("World: {}/{}".format(wrank, wsize))
if wrank == 0:
  logging.info(type(ans))

#from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

x = [1, 2, 3, 4]
colour = rank%2
comm2 = comm.Split(color=colour, key=rank)

data = x[comm2.Get_rank()]

data = comm.reduce(data, root=0)
print(comm.Get_rank(), comm2.Get_rank(), data)

"""
"""
comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

#pcg = PCG64(seed=123456789)
#rg = RandomGenerator(pcg)

levels = 3
reps = [12, 6, 2]

results = [0 for i in range(levels)]
comms = [comm]
new_reps = [2]

for i in range(levels-1):
  color = comms[0].Get_rank() % 2
  new_comm = comms[0].Split(color=color)

  ratio = comms[-1].Get_size() / new_comm.Get_size()

  comms.insert(0, new_comm)
  new_reps.insert(0, int(reps[-(i+2)]/ratio))


print(new_reps)
for i in range(levels):
  mesh = UnitSquareMesh(10, 10, comm=comms[i])
  V = FunctionSpace(mesh, "Lagrange", 4)
  subresults = []

  for j in range(new_reps[i]):
    uh1 = matern(V, mean=1, variance=0.2, correlation_length=0.1, smoothness=1)
    #uh1 = rg.beta(V, 1.0, 2.0)
    ans = assemble(Constant(0.5) * dot(uh1, uh1) * dx)
    subresults.append(ans)
  
  results[i] = sum(subresults)/len(subresults)

y = np.array([0, 0, 0], dtype=np.double)
comm.Reduce([np.array(results), MPI.DOUBLE], [y, MPI.DOUBLE], op=MPI.SUM, root=0)

print(results)    
print(y)
"""
"""
logger = logging.logging.getLogger("MLMC-logger")
logger.setLevel(logging.INFO)
file_handler = logging.logging.StreamHandler()
logger.addHandler(file_handler)


comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

x = np.array([1, 2, 3, 4])

color = world_rank % 2

row_comm = comm.Split(color=color)

print(row_comm)
row_rank = row_comm.Get_rank()
row_size = row_comm.Get_size()
comm.Barrier()
logger.error("WORLD RANK/SIZE: {}/{} \t ROW RANK/SIZE: {}/{}\n".format(
	world_rank, world_size, row_rank, row_size))
comm.Barrier()
if comm.rank == 0:
    y = np.array([0, 0, 0, 0])
else:
    y = None
comm.Reduce([x, MPI.DOUBLE], [y, MPI.DOUBLE], op=MPI.SUM, root=0)
print(y)
#comm = MPI.COMM_WORLD
#help(comm.Reduce)

"""


logger = logging.logging.getLogger("MLMC-logger")
logger.setLevel(logging.INFO)
file_handler = logging.logging.StreamHandler()
logger.addHandler(file_handler)

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

levels = 2
repetitions = [1, 1]

results = [0 for i in range(levels)]
comms = [comm]
new_reps = [repetitions[-1]]
did_calculate = [1]

colour_list = []
for i in range(levels-1):
    colour = comms[0].Get_rank() % 2
    colour_list.append(colour)
    new_comm = comms[0].Split(color=colour)

    num_comms = comms[-1].Get_size() / new_comm.Get_size()
    comms.insert(0, new_comm)

    rep = repetitions[-(i+2)]//num_comms
    rep_r = repetitions[-(i+2)]%num_comms
    # Find decimal number of colour_num binary list
    colour_num = int("".join(map(str, colour_list)),2)
    
    # decimal number found should always be less than num_comms
    assert num_comms == 2**len(colour_list), \
    ("Communicator Division Error: Must be an even number of cores in COMM_WORLD")
    # Distribute remainder across cores
    if colour_num < rep_r:
        rep += 1
    new_reps.insert(0, int(rep))
    
    # If no reps performed make not to ensure the correct average is taken
    if rep == 0:
        did_calculate.insert(0, 0)
    else:
        did_calculate.insert(0, 1)

for i in range(levels):
    if new_reps[i] != 0:
        mesh = UnitSquareMesh(10, 10, comm=comms[i])
        V = FunctionSpace(mesh, "Lagrange", 4)
    
    logger.info("new_level - level: {}".format(i))
    subresults = []
    for j in range(new_reps[i]):
        #uh1 = rg.beta(V, 1.0, 2.0)
        uh1 = matern(V, mean=1, variance=0.2, correlation_length=0.1, smoothness=1)
        ans = assemble(Constant(0.5) * dot(uh1, uh1) * dx)
        subresults.append(ans)
    
    if len(subresults) == 0:
        results[i] = 0
    else:
        results[i] = sum(subresults)/len(subresults)

y = np.zeros_like(results, dtype=np.float64) 
comm.Reduce([np.array(results), MPI.DOUBLE], [y, MPI.DOUBLE], op=MPI.SUM, root=0)

print(list(results))    
print(list(y))
"""
logger = logging.logging.getLogger("MLMC-logger")
logger.setLevel(logging.INFO)
file_handler = logging.logging.StreamHandler()
logger.addHandler(file_handler)

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

colour = comm.Get_rank() % 2
new_comm = comm.Split(color=colour)

pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

comms = [new_comm, comm]
levels = 2
if colour == 0:
    new_reps = [0, 1]
else:
    new_reps = [1, 1]

results = [0, 0]

mesh = UnitSquareMesh(10, 10, comm=comms[0])
V = FunctionSpace(mesh, "Lagrange", 4)

subresults = []

if new_reps[0] == 1:
    #uh1 = rg.beta(V, 1.0, 2.0)
    uh1 = matern(V, mean=1, variance=0.2, correlation_length=0.1, smoothness=1)
    ans = assemble(Constant(0.5) * dot(uh1, uh1) * dx)
    results[0] = ans

else:
    results[0] = 0

mesh = UnitSquareMesh(10, 10, comm=comms[1])
V = FunctionSpace(mesh, "Lagrange", 4)
uh1 = matern(V, mean=1, variance=0.2, correlation_length=0.1, smoothness=1)
#uh1 = rg.beta(V, 1.0, 2.0)
logger.info("got here")
ans = assemble(Constant(0.5) * dot(uh1, uh1) * dx)
logger.info("not here")
results[1] = ans


y = np.zeros_like(results, dtype=np.float64) 
comm.Reduce([np.array(results, dtype=np.float64), MPI.DOUBLE], [y,MPI.DOUBLE], op=MPI.SUM, root=0)

logger.info("results and rank {} - {}".format(results, world_rank))
logger.info("sum and rank {} - {}".format(y, world_rank))
"""
