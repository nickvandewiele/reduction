
#global imports
import itertools
import copy
import os.path
import csv
import numpy as np
from math import ceil
from scipy.optimize import minimize
import re
import random
from collections import Counter
import logging

#local imports

global useSCOOP
useSCOOP = False

try:
    from scoop import shared
    from scoop.futures import map as map_
    from scoop import logger as logging
except ImportError:
    logging.error('Import Error!')
    map_ = map
    import logging

from rmgpy.chemkin import getSpeciesIdentifier, loadChemkinFile
from rmgpy.rmg.main import RMG
from rmgpy.solver.base import TerminationTime, TerminationConversion

"""
Guidelines for input:

- Do not use the annotated chemkin file as input!
- 
"""

CLOSE_TO_ZERO = 1E-10

class ReductionReaction(object):
    """
    A class that enhances RMG-Py's  Reaction object
    by providing storage for the forward (kf) and backward
    (kb) rate coefficient.

    Once k is computed, it is stored and fetched
    when requested.

    """
    def __init__(self, rmg_reaction):
        super(ReductionReaction, self).__init__()
        self.rmg_reaction = rmg_reaction
        self.reactants = rmg_reaction.reactants
        self.products = rmg_reaction.products
        self.kf = None
        self.kb = None
        self.stoichio = {}
        self.create_stoichio()
    
    def __str__(self):
        return str(self.rmg_reaction)

    def __reduce__(self):
        """
        A helper function used when pickling an object.
        """
        return (self.__class__, (self.rmg_reaction, ))


    def getRateCoefficient(self, T,P):
        if self.kf is None:
            self.kf = self.rmg_reaction.getRateCoefficient(T,P)
            return self.kf
        else: return self.kf
    
    def getReverseRateCoefficient(self, T, P):
        if self.kb is None:
            kf = self.getRateCoefficient(T,P) 
            self.kb = kf / self.rmg_reaction.getEquilibriumConstant(T)
            return self.kb
        else: return self.kb
    
    def create_stoichio(self):
        c_reactants = Counter([mol.label for mol in self.reactants])
        self.stoichio['reactant'] = c_reactants

        c_products = Counter([mol.label for mol in self.products])
        self.stoichio['product'] = c_products

    def get_stoichiometric_coefficient(self, spc_i, reactant_or_product):       
        return self.stoichio[reactant_or_product][spc_i.label]


def simulate_one(reactionModel, atol, rtol, reactionSystem):
    """

    Simulates one reaction system, listener registers results, 
    which are returned at the end.


    The returned data consists of a array of the species names, 
    and the concentration data.

    The concentration data consists of a number of elements for each timestep 
    the solver took to reach the end time of the batch reactor simulation.

    Each element consists of the time and the concentration data of the species at that 
    particular timestep in the order of the species names.

    """

    #register as a listener
    listener = ConcentrationListener()

    coreSpecies = reactionModel.core.species
    regex = r'\([0-9]+\)'#cut of '(one or more digits)'
    species_names = []
    for spc in coreSpecies:
        name = getSpeciesIdentifier(spc)
        name_cutoff = re.split(regex, name)[0]
        species_names.append(name_cutoff)

    listener.species_names = species_names

    reactionSystem.attach(listener)

    pdepNetworks = []
    for source, networks in reactionModel.networkDict.items():
        pdepNetworks.extend(networks)
    
    terminated, obj = reactionSystem.simulate(
        coreSpecies = reactionModel.core.species,
        coreReactions = reactionModel.core.reactions,
        edgeSpecies = reactionModel.edge.species,
        edgeReactions = reactionModel.edge.reactions,
        toleranceKeepInEdge = 0,
        toleranceMoveToCore = 1,
        toleranceInterruptSimulation = 1,
        pdepNetworks = pdepNetworks,
        absoluteTolerance = atol,
        relativeTolerance = rtol,
    ) 

    assert terminated

    #unregister as a listener
    reactionSystem.detach(listener) 

    return listener.species_names, listener.data

def simulate_all(rmg):
    """
    Simulate the RMG job, generating an output csv file
    for each of the simulated reaction systems.

    Each element i of the data corresponds to a reaction system.
    """
    
    reactionModel = rmg.reactionModel

    data = []

    atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
    for _, reactionSystem in enumerate(rmg.reactionSystems):
        data.append(simulate_one(reactionModel, atol, rtol, reactionSystem))

    return data
        


def initialize(wd):
    global working_dir
    working_dir = wd
    assert os.path.isdir(working_dir)
    

def find_unimportant_reactions(rxns, rmg, tolerance):
    """
    This function:

    - loops over all rxns
    - loops over all the species involved in a specific reaction
    - decides whether the specific reaction is important for the species.

    Whenever it is found that a reaction is important for a species, we break
    the species loop, and keep the reaction in the model.


    Returns:
        a list of rxns that can be removed.
    """

    # run the simulation, creating csv concentration profiles for each reaction system defined in input.
    simdata = simulate_all(rmg)
    if useSCOOP:
        shared.setConst(data = simdata)
    else:
        global data
        data = simdata
    # logging.info('Sharing data: {}'.format(data))


    # start the model reduction
    reduce_reactions = [ReductionReaction(rxn) for rxn in rxns]
    if useSCOOP:
        shared.setConst(reactions = reduce_reactions)
    else:
        global reactions
        reactions = reduce_reactions
    

    """
    Tolerance to decide whether a reaction is unimportant for the formation/destruction of a species

    Tolerance is a floating point value between 0 and 1.

    A high tolerance means that many reactions will be deemed unimportant, and the reduced model will be drastically
    smaller.

    A low tolerance means that few reactions will be deemed unimportant, and the reduced model will only differ from the full
    model by a few reactions.
    """

    N = len(reduce_reactions)
    boolean_array = list(map_(assess_reaction, reduce_reactions, [rmg.reactionSystems] * N, [tolerance] * N))

    reactions_to_be_removed = []
    for isImport, rxn in zip(boolean_array, reduce_reactions):
        if not isImport:
            reactions_to_be_removed.append(rxn)


    return [rxn.rmg_reaction for rxn in reactions_to_be_removed]
    
    return myfilter

def mock_assess_reaction(rxn, reactionSystems, tolerance):
    return bool(random.getrandbits(1))

def assess_reaction(rxn, reactionSystems, tolerance):
    """
    Returns whether the reaction is important or not in the reactions.

    It iterates over the reaction systems, and loads the concentration profile 
    of each reaction system into memory.

    It iterates over a number of samples in profile and 
    evaluates the importance of the reaction at every sample.


    """
    logging.info('Assessing reaction {}'.format(rxn))
    if useSCOOP:
        reactions = shared.getConst('reactions')
        data = shared.getConst('data')
    else:
        global data, reactions


    # read in the intermediate state variables
    for datum, reactionSystem in zip(data, reactionSystems):    
        T, P = reactionSystem.T.value_si, reactionSystem.P.value_si
        
        species_names, profile = datum

        # take N evenly spaced indices from the table with simulation results:

        """

        Number of time steps between start and end time of the batch reactor simulation at which the importance of 
        reactions should be evaluated.



        The more timesteps, the less chance we have to remove an important reactions, but the more simulations
        need to be carried out.
        """
        
        timesteps = len(profile) / 4
        logging.debug('Evaluating the importance of a reaction at {} time samples.'.format(timesteps))

        assert timesteps <= len(profile)
        indices = map(int, np.linspace(0, len(profile)-1, num = timesteps))
        for index in indices:
            assert profile[index] is not None
            timepoint, coreSpeciesConcentrations = profile[index]

            coreSpeciesConcentrations = {key: float(value) for (key, value) in zip(species_names, coreSpeciesConcentrations)}
            
            # print 'Species concentrations at {}: {}'.format(timepoint, reactionSystem.coreSpeciesConcentrations)
            for species_i in rxn.reactants:
                if isImportant(rxn, species_i, reactions, 'reactant', tolerance, T, P, coreSpeciesConcentrations):
                    return True

            #only continue if the reaction is not important yet.
            for species_i in rxn.products:
                if isImportant(rxn, species_i, reactions, 'product', tolerance, T, P, coreSpeciesConcentrations):
                    return True

    return False



def isImportant(rxn, species_i, reactions, reactant_or_product, tolerance, T, P, coreSpeciesConcentrations):
    """
    This function computes:
    - Ri = R(species_i)
    - rij = r(rxn)
    - alpha = ratio of rij / Ri
    
    Range of values of alpha:
    0 <= alpha <= 1

    This function also compares alpha to a user-defined tolerance TOLERANCE.
    if alpha >= tolerance:
        this reaction is important for this species.
    else:
        this reaction is unimportant for this species.

    Returns whether or not rxn is important for species_i.
    keep = True
    remove = False
    """
    #calc Ri, where i = species


    rij = calc_rij(rxn, species_i, reactant_or_product, T, P, coreSpeciesConcentrations) 
    Ri = calc_Ri(species_i, rij, reactions, reactant_or_product, T, P, coreSpeciesConcentrations)

    # assert Ri != 0, "rij: {0}, Ri: {1}, rxn: {2}, species: {3}, reactant: {4}"\
    # .format(rij, Ri, rxn, species_i, reactant_or_product)

    # if rij == 0  and Ri == 0:
    
    if np.any(np.absolute([rij, Ri]) < CLOSE_TO_ZERO):
       # print "rij: {0}, Ri: {1}, rxn: {2}, species: {3}, reactant: {4}, alpha: {5}, tolerance: {6}"\
       #  .format(rij, Ri, rxn, species_i, reactant_or_product, 'N/A', tolerance) 
       return False

    else:
        assert Ri != 0, "rij: {0}, Ri: {1}, rxn: {2}, species: {3}, reactant: {4}".format(rij, Ri, rxn, species_i, reactant_or_product)
        alpha = rij / Ri
        if alpha < 0: return False


    if alpha > tolerance :
        """
        If both values are very close to 1, then the comparison of alpha and the tolerance
        might sometimes return an unexpected value.

        When we set the tolerance to a value of 1, we want all the reactions to be unimportant,
        regardless of the value of alpha.

        """
        if np.allclose([tolerance, alpha], [1.0, 1.0]):
            return False
            
        # print "rij: {0}, Ri: {1}, rxn: {2}, species: {3}, reactant: {4}, alpha: {5}, tolerance: {6}"\
        # .format(rij, Ri, rxn_j, species_i, reactant_or_product, alpha, tolerance)
        return True
        #where tolerance is user specified tolerance
 
    elif alpha <= tolerance:
        return False
    

def compute_reaction_rate(rxn_j, forward, T, P, coreSpeciesConcentrations): 
    """

    Computes reaction rate r as follows:

    r = k * Product(Ci^nu_ij, for all j)
    with:

    k = rate coefficient for rxn_j,
    Cij = the concentration for molecule i ,
    nu_ij = the stoichiometric coefficient for molecule i in reaction j.

    ...
    """

    k = rxn_j.getRateCoefficient(T,P) if forward else rxn_j.getReverseRateCoefficient(T,P)
    species_list = rxn_j.reactants if forward else rxn_j.products
    reactant_or_product = 'reactant' if forward else 'product'

    assert species_list is not None

    concentrations = []
    for spc_i in species_list:
        ci = getConcentration(spc_i, coreSpeciesConcentrations)
        nu_i = rxn_j.get_stoichiometric_coefficient(spc_i, reactant_or_product)
        concentrations.append(ci**nu_i)

    
    product = 1
    for conc in concentrations:
        if np.absolute(conc) < CLOSE_TO_ZERO:
            return 0.
        product = product * conc
        #print 'The product of conc raised to stoich coeff is: ', product

    r = k * product


    return r


def getConcentration(spc, coreSpeciesConcentrations):
    """
    Returns the concentration of the species in the 
    reaction system.
    """
    return coreSpeciesConcentrations[spc.label]

def calc_rij(rxn_j, spc_i, reactant_or_product, T, P, coreSpeciesConcentrations):
    """
    This function computes the rate of formation of species i
    through the reaction j.

    This function multiplies:
    - nu(i): stoichiometric coefficient of spc_i in rxn_j
    - r(rxn_j): reaction rate of rxn_j

    Returns a reaction rate

    Units: mol / m^3 s
    """
   
    nu_i = rxn_j.get_stoichiometric_coefficient(spc_i, reactant_or_product)
    sign = -1 if reactant_or_product == 'reactant' else 1

    forward = reactant_or_product == 'reactant'

    r_j = compute_reaction_rate(rxn_j, forward, T, P, coreSpeciesConcentrations)

    rij = nu_i * sign * r_j
    return rij


def calc_Rf(spc_i, reactions, reactant_or_product, T, P, coreSpeciesConcentrations, formation_or_consumption):
    """
    Calculates the total rate of formation/consumption of species i.

    Computes the sum of the rates of formation/consumption of spc_i for all of 
    the reactions in which spc_i is a product. 

    if formation_or_consumption == 'formation', spc_i will be compared to the 
    products of the reaction. Else, spc_i will be compared to the reactants of
    the reaction.

    units of rate: mol/(m^3.s)
    """
    rate = 0.0

    for reaction in reactions:
        molecules = reaction.products if formation_or_consumption == 'formation:' else reaction.reactants
        labels = [mol.label for mol in molecules]
        if spc_i.label in labels:
            rij = calc_rij(reaction, spc_i,  reactant_or_product, T, P, coreSpeciesConcentrations)
            rate = rate + rij


    return rate
    
def calc_Rf_closure(spc_i, reactions, reactant_or_product, T, P, coreSpeciesConcentrations):
    """
    Closure to avoid replicating function calls to calc_Rf.
    """
    def myfilter(formation_or_consumption):
        return calc_Rf(spc_i, reactions, reactant_or_product, T, P, coreSpeciesConcentrations, formation_or_consumption)
    
    return myfilter

def calc_Ri(spc_i,rij, reactions, reactant_or_product, T, P, coreSpeciesConcentrations):
    """

    Checks whether the sign of rij to decide to compute the
    total rate of formation or consumption of spc_i.

    units of rate: mol/(m^3.s)
    """

    f_closure = calc_Rf_closure(spc_i, reactions, reactant_or_product, T, P, coreSpeciesConcentrations)

    if rij > 0:
        return f_closure('formation')
    elif rij < 0:
        return f_closure('consumption') 
    elif np.absolute([rij]) < CLOSE_TO_ZERO:
        """Pick the largest value so that the ratio rij / RX remains small."""
        Rf = f_closure('formation')
        Rb = f_closure('consumption')

        """What happens when Rf ~ Rb <<< 1?"""
        return max(abs(Rf),abs(Rb))

def print_info(rxn, spc, important):
    logging.info('Is reaction {0} important for species {1}: {2}'.format(rxn, spc, important))
 

def loadRMGPyJob(inputFile, chemkinFile, speciesDict=None):
    """
    Load the results of an RMG-Py job generated from the given `inputFile`.
    """
    
    # Load the specified RMG input file
    rmg = RMG()
    rmg.loadInput(inputFile)
    rmg.outputDirectory = os.path.abspath(os.path.dirname(inputFile))
    
    # Load the final Chemkin model generated by RMG
    speciesList, reactionList = loadChemkinFile(chemkinFile, speciesDict, readComments=False)
    assert speciesList, reactionList

    # print 'labels from species in the chemkin file:'
    # for spc in speciesList:
    #     print spc.label

    # Map species in input file to corresponding species in Chemkin file
    speciesDict = {}
    assert rmg.initialSpecies
    # print 'initial species: ', rmg.initialSpecies


    #label of initial species must correspond to the label in the chemkin file WITHOUT parentheses.
    #E.g.: "JP10" not "JP10(1)"
    for spec0 in rmg.initialSpecies:
        for species in speciesList:
            if species.label == spec0.label:
                speciesDict[spec0] = species
                break
            
    assert speciesDict
    # Replace species in input file with those in Chemkin file
    for reactionSystem in rmg.reactionSystems:
        reactionSystem.initialMoleFractions = dict([(speciesDict[spec], frac) for spec, frac in reactionSystem.initialMoleFractions.iteritems()])
        for t in reactionSystem.termination:
            if isinstance(t, TerminationConversion):
                t.species = speciesDict[t.species]
    
    # Set reaction model to match model loaded from Chemkin file
    rmg.reactionModel.core.species = speciesList
    rmg.reactionModel.core.reactions = reactionList

    # print 'core: ', speciesList   
    return rmg

def write_model(rmg, chemkin_name='reduced_reactions.inp'):
    saveChemkinFile(chemkin_name, rmg.reactionModel.core.species, rmg.reactionModel.core.reactions)

def remove_reactions_from_model(rmg, reaction_list):
    """
    Assumes reactions are in the core:
    """

    new_core_rxns = []
    for rxn in rmg.reactionModel.core.reactions:
        if not rxn in reaction_list:
            new_core_rxns.append(rxn)

    rmg.reactionModel.core.reactions = new_core_rxns
    return rmg

def saveChemkinFile(path, species, reactions, verbose = True):
    from rmgpy.chemkin import writeKineticsEntry

    s ='REACTIONS    KCAL/MOLE   MOLES\n\n'

    for rxn in reactions:
        s +=writeKineticsEntry(rxn, speciesList=species, verbose=verbose)
        s +='\n'
    s +='END\n\n'

    with open(path, 'w') as f:
        f.write(s)

def compute_conversion(target, reactionModel, reactionSystem, reactionSystem_index, atol, rtol):
    """
    Computes the conversion of a target molecule by

    - searching the index of the target species in the core species
    of the global reduction variable
    - resetting the reaction system, initialing with empty variables
    - fetching the initial moles variable y0
    - running the simulation at the conditions stored in the reaction system
    - fetching the computed moles variable y
    - computing conversion
    """

    target_index = reactionModel.core.species.index(target)

    #reset reaction system variables:
    reactionSystem.initializeModel(\
        reactionModel.core.species, reactionModel.core.reactions,\
        reactionModel.edge.species, reactionModel.edge.reactions, \
        [], atol, rtol)

    #get the initial moles:
    y0 = reactionSystem.y.copy()

    #run the simulation:
    simulate_one(reactionModel, atol, rtol, reactionSystem)

    #compute conversion:
    conv = 1 - (reactionSystem.y[target_index] / y0[target_index])
    return conv

def reduce_compute(tolerance, target, reactionModel, rmg, reaction_system_index):
    """
    Reduces the model for the given tolerance and evaluates the 
    target conversion.
    """

    # reduce model with the tolerance specified earlier:
    reactions_to_be_removed = find_unimportant_reactions(reactionModel.core.reactions, rmg, tolerance)

    original_size = len(reactionModel.core.reactions)
    logging.info('Initial model size: {}'.format(original_size))

    no_unimportant_rxns = len(reactions_to_be_removed)
    logging.info('Number of unimportant reactions: {}'.format(no_unimportant_rxns))

    # remove reactions from core:
    original_reactions = reactionModel.core.reactions
    remove_reactions_from_model(rmg, reactions_to_be_removed)

    #re-compute conversion: 
    conversion = compute_conversion(target, rmg.reactionModel,\
     rmg.reactionSystems[reaction_system_index], reaction_system_index,\
     rmg.absoluteTolerance, rmg.relativeTolerance)

    #reset the reaction model to its original state:
    rmg.reactionModel.core.reactions = original_reactions

    logging.info('Conversion of reduced model ({} rxns): {:.2f}%'.format(original_size - no_unimportant_rxns, conversion * 100))
    return conversion

def objective(tolerance, target, reactionModel, rmg, reaction_system_index, allowed_error, Xorig):
    """
    Objective function to be minimized as a function of the reduction tolerance
    with x the reduction tolerance.

    The reduction tolerance is used to compute a reduced model. For the reduced model, 
    the conversion of the target parameter is computed and compared to the conversion
    of the original Xorig, non-reduced model.

    The deviation between the conversions of the original and reduced model is computed:
    dev = (Xred - Xorig) / Xorig

    The function f to be minimized is taken as:
    f = dev^2 - allowed_error^2

    0 < x < 1
    
    """
    tolerance = tolerance[0]
    Xred = reduce_compute(tolerance, target, reactionModel, rmg, reaction_system_index)
    
    dev = (Xred - Xorig) / Xorig
    logging.info('Deviation between original and reduced conversion: {:.2f}%'.format(dev * 100))

    scale = 1e3 #rescale your objective function so that the differences and derivatives are larger
    f = (dev * dev - allowed_error * allowed_error) * scale

    logging.info('Objective function value: {:.2f}'.format(f))

    # derror = dconv(x)
    # df = 2 * dev * derror
    return f

def callback_x(x):
    logging.info('Current parameter vector: {}'.format(x))

def optimize_tolerance(target, reactionModel, rmg, reaction_system_index, error, orig_conv):
    """
    Unconstrained minimization with bounds on the variable x.
    """
    x0 = 1e-3#initial guess
    logging.info('Initial guess for the reduction tolerance: {}'.format(x0))

    res = minimize(objective, np.array([x0]),\
     args=(target, reactionModel, rmg, reaction_system_index, error, orig_conv),\
     bounds=[(0,1)], tol=1e-8, options={'disp': True},callback = callback_x)#method='nelder-mead',\
     
    logging.info(res)
    return res.x

class ConcentrationListener(object):
    """Returns the species concentration profiles at each time step."""

    def __init__(self):
        self.species_names = []
        self.data = []

    def update(self, subject):
        self.data.append((subject.t , subject.coreSpeciesConcentrations))