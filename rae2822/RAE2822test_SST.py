# ======================================================================
#         Import modules
# ======================================================================
# rst Imports (beg)
import sys,os,traceback
import numpy as np
import argparse
import time
from mpi4py import MPI
from baseclasses import AeroProblem
from adflow_sst import ADFLOW
from multipoint import multiPointSparse
from collections import OrderedDict
import pprint
import copy

# rst Imports (end)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--gridFile", type=str, default="rae2822L2.cgns")
args = parser.parse_args()
# ======================================================================
#         Create multipoint communication object
# ======================================================================
# rst multipoint (beg)
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
os.makedirs(args.output,exist_ok = True)
# rst multipoint (end)

# ======================================================================
#         Specify parameters for caculation
# ======================================================================
# rst parameters (beg)
# angle of attack
alphamean = 2.79
# mach number
machmean = 0.734
# reynold number
reynolds = 6.5e6
reynoldsLength = 1.0
# reference area
areaRef = 1.0
# reference chord
chordRef = 1.0
# temperature K
T = 223.252
# rst parameters (end)
grid_file ='./'+args.gridFile

# ======================================================================
#         ADflow Set-up
# ======================================================================
# rst adflow (beg)
aeroOptions = {
    # I/O Parameters
    'gridFile':grid_file,
    'outputDirectory':args.output,
    'monitorvariables':['resrho','totalr','cl','cpu','cd','cmz','resturb','cdp','cdv'],
    'volumevariables':['resrho','Intermittency', 'cp', 'mach', 'temp', 'rhoe'],
    'surfacevariables':['cp','vx', 'vy','vz', 'mach','cfx', 'mach', 'rho', 'p', 'temp', 'cf', 'yplus','blank'],
    'writeTecplotSurfaceSolution':True,
	# 'solRestart' : True,
	#'restartFile' : './fc_000_vol.cgns',

    # Physics Parameters
    'equationType':'RANS',
    'turbulenceModel':'Menter SST', 
    'use2003SST':True,
    'smoothSSTphi':[1e3,1e1,1e15,1e3,1e4],  
    'turbresscale':[1e3,1e-6],
    #'turbResScale': 10000.0, #for SA
  
    # Solver Parameters
    'CFL':1.5,
    'CFLCoarse':1.25,
    'MGCycle':'SG',
  
    # Termination Criteria
    'L2Convergence':1e-12,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':30000,
    'useblockettes':False,
  
    'rkreset': True,
    'nrkreset' : 100,
    'smoother':'Runge-Kutta',
    'resaveraging':'alternate',
    'smoothparameter':1.5, 



    # ANK Solver Parameters
    'useANKSolver':True,
    'ANKADPC':True, #better preconditioner 
    'ANKPCILUFill': 3,  
    'anklinearsolvetol':0.1,
    'nsubiterturb':15,
    'ankswitchtol':1e-2,
    'anksecondordswitchtol':1e-6, # increased for 30 deg
	'ankstepfactor' : 0.5,
    'anksubspacesize':300,
	#'ankcoupledswitchtol' : 1e-6,
	'ankmaxiter' : 60,
    'nsubiterturb':40,
  
 


    
    #Turb stuff
    # 'useqcr':True, # go 2 NASA tmr for more info (closer to exp??)
    # Following 3 give defalt SA-noft2 (fully turb sim, in lit)
    # 'eddyvisinfratio':.210438,
    'useft2SA' : False,
    # 'turbulenceproduction' : 'vorticity',
    
    # NK Solver Parameters
    'useNKSolver':True,
    'nkswitchtol':1e-9,
    'nkadpc':True,
    # 'nkasmoverlap': 3, # for highly parallel
    'NKASMOverlap': 4,
    'NKInnerPreconIts': 3,
    'NKJacobianLag': 5,
    'NKOuterPreconIts': 3,
    'NKPCILUFill': 4,
    'NKSubspaceSize': 400,


    # # Adjoint Parameters
    'setMonitor':True,
    "adjointSolver": "GMRES",
    "adjointL2Convergence": 1e-12,
    "ADPC": True,
    "adjointMaxIter": 5000,
    "adjointSubspaceSize": 400,
    "ILUFill": 3,
    "ASMOverlap": 3,
    "outerPreconIts": 3,
    "frozenTurbulence": False,
    "restartADjoint": False,#todo? but it's for the adjoint
}

# Create solver
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
span = 1
pos  = np.array([0.5])*span
CFDSolver.addSlices('z',pos,sliceType='absolute')

# rst adflow (end)

# ======================================================================
#         Set up flow conditions with AeroProblem
# ======================================================================
# rst aeroproblem (beg)
name = 'RAE2822'
aeroProblems = []


    
ap = AeroProblem(name=name,
        mach=machmean,
        reynolds=reynolds,
        reynoldsLength = reynoldsLength,
        T = T ,
        alpha=alphamean,
        areaRef=areaRef,
        chordRef=chordRef,
        xRef=0.25,yRef=0.0,zRef=0.0,
        evalFuncs=['cl','cd','cmz'])
aeroProblems.append(ap)

# rst aeroproblem (end)


# ======================================================================
#         Solve Functions:
# ======================================================================
StartTime = time.time()
                 
for i in range(len(aeroProblems)):
    funcs = {}
    sol = {}
    CFDSolver(aeroProblems[i])
    CFDSolver.evalFunctions(aeroProblems[i], funcs)
    sol = CFDSolver.getSolution()
    if MPI.COMM_WORLD.rank == 0:
        print(f"{funcs=}")
        print(f"{sol=}")


EndTime = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f"time used = {EndTime-StartTime}")