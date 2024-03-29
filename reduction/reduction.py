
#global imports
import copy
import os.path
import numpy as np
import re
from collections import Counter
import sys

#local imports

#global variables
reactions = None

try:
    from scoop import shared
    from scoop.futures import map as map_
    from scoop import logger as logging
    logging.setLevel(20)#10 : debug, 20: info
except ImportError:
    map_ = map
    import logging
    logging.error('Import Error!')

from rmgpy.chemkin import getSpeciesIdentifier, loadChemkinFile
from rmgpy.rmg.main import RMG
from rmgpy.solver.base import TerminationTime, TerminationConversion
from workerwrapper import WorkerWrapper
"""
Guidelines for input:

- Do not use the annotated chemkin file as input!
- 
"""

CLOSE_TO_ZERO = 1E-20

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
    Simulate the RMG job, 
    for each of the simulated reaction systems.

    Each element i of the data corresponds to a reaction system.
    """
    reactionModel = rmg.reactionModel

    data = []

    atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
    for _, reactionSystem in enumerate(rmg.reactionSystems):
        data.append(simulate_one(reactionModel, atol, rtol, reactionSystem))

    return data
        


def initialize(wd, rxns):
    global working_dir, reactions
    working_dir = wd
    assert os.path.isdir(working_dir)
    
    #set global variable here such that functions executed in the root worker have access to it.
    reactions = rxns
    reduce_reactions = [ReductionReaction(rxn) for rxn in rxns]
    shared.setConst(reactions = reduce_reactions)


def find_important_reactions(rmg, tolerance):
    """
    This function:

    - loops over all the species involved in a specific reaction
    - decides whether the specific reaction is important for the species.

    Whenever it is found that a reaction is important for a species, we break
    the species loop, and keep the reaction in the model.


    Returns:
        a list of rxns that can be removed.
    """

    global reactions
    
    # run the simulation, creating concentration profiles for each reaction system defined in input.
    simdata = simulate_all(rmg)

    reduce_reactions = [ReductionReaction(rxn) for rxn in reactions]

    """
    Tolerance to decide whether a reaction is unimportant for the formation/destruction of a species

    Tolerance is a floating point value between 0 and 1.

    A high tolerance means that many reactions will be deemed unimportant, and the reduced model will be drastically
    smaller.

    A low tolerance means that few reactions will be deemed unimportant, and the reduced model will only differ from the full
    model by a few reactions.
    """

    CHUNKSIZE = 40
    boolean_array = []
    for chunk in chunks(reduce_reactions,CHUNKSIZE):
        N = len(chunk)
        partial_results = list(map_(WorkerWrapper(assess_reaction), chunk, [rmg.reactionSystems] * N, [tolerance] * N, [simdata] * N))
        boolean_array.extend(partial_results)

    important_rxns = []
    for isImport, rxn in zip(boolean_array, reduce_reactions):
        logging.debug('Is rxn {rxn} important? {isImport}'.format(**locals()))
        if isImport:
            important_rxns.append(rxn)


    return [rxn.rmg_reaction for rxn in important_rxns]

def assess_reaction(rxn, reactionSystems, tolerance, data):
    """
    Returns whether the reaction is important or not in the reactions.

    It iterates over the reaction systems, and loads the concentration profile 
    of each reaction system into memory.

    It iterates over a number of samples in profile and 
    evaluates the importance of the reaction at every sample.


    """
    
    logging.debug('Assessing reaction {}'.format(rxn))

    reactions = shared.getConst('reactions')
    

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
        
        timesteps = len(profile) / 2
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

    logging.debug("rij: {rij}, Ri: {Ri}, rxn: {rxn}, species: {species_i}, reactant: {reactant_or_product}, tol: {tolerance}"\
    .format(**locals()))

    if np.any(np.absolute([rij, Ri]) < CLOSE_TO_ZERO):
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
    isReactant = forward

    assert species_list is not None

    concentrations = np.empty(len(species_list), dtype=float)
    for i,spc_i in enumerate(species_list):
        ci = getConcentration(spc_i, coreSpeciesConcentrations)
        if np.absolute(ci) < CLOSE_TO_ZERO:
            return 0.
        nu_i = rxn_j.get_stoichiometric_coefficient(spc_i, isReactant)
        conc = ci**nu_i

        concentrations[i] = conc

    r = k * concentrations.prod()


    return r


def getConcentration(spc, coreSpeciesConcentrations):
    """
    Returns the concentration of the species in the 
    reaction system.
    """
    return coreSpeciesConcentrations[spc.label]

def calc_rij(rxn_j, spc_i, isReactant, T, P, coreSpeciesConcentrations):
    """
    This function computes the rate of formation of species i
    through the reaction j.

    This function multiplies:
    - nu(i): stoichiometric coefficient of spc_i in rxn_j
    - r(rxn_j): reaction rate of rxn_j

    Returns a reaction rate

    Units: mol / m^3 s
    """
   
    nu_i = rxn_j.get_stoichiometric_coefficient(spc_i, isReactant)
    sign = -1 if isReactant else 1

    forward = isReactant

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

    logging.debug('Rf: {rate}'.format(**locals()))

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


def saveChemkinFile(path, species, reactions, verbose = True):
    from rmgpy.chemkin import writeKineticsEntry

    s ='REACTIONS    KCAL/MOLE   MOLES\n\n'

    for rxn in reactions:
        s +=writeKineticsEntry(rxn, speciesList=species, verbose=verbose)
        s +='\n'
    s +='END\n\n'

    with open(path, 'w') as f:
        f.write(s)

def search_target(target_label, reactionSystem):

    for k in reactionSystem.initialMoleFractions.keys():
        if k.label == target_label:
            target = k
            break
    assert target is not None, '{} could not be found...'.format(target_label)
    return target


def compute_conversion(target_label, reactionModel, reactionSystem, reactionSystem_index, atol, rtol):
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

    target = search_target(target_label, reactionSystem)
    target_index = reactionModel.core.species.index(target)

    #reset reaction system variables:
    logging.info('No. of rxns in core reactions: {}'.format(len(reactionModel.core.reactions)))
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

def reduce_compute(tolerance, target_label, reactionModel, rmg, reaction_system_index):
    """
    Reduces the model for the given tolerance and evaluates the 
    target conversion.
    """

    # reduce model with the tolerance specified earlier:
    important_reactions = find_important_reactions(rmg, tolerance)

    original_size = len(reactionModel.core.reactions)

    no_important_reactions = len(important_reactions)
    logging.info('Number of important reactions: {}'.format(no_important_reactions))

    #set the core reactions to the reduced reaction set:
    original_reactions = reactionModel.core.reactions
    rmg.reactionModel.core.reactions = important_reactions

    #re-compute conversion: 
    conversion = compute_conversion(target_label, rmg.reactionModel,\
     rmg.reactionSystems[reaction_system_index], reaction_system_index,\
     rmg.absoluteTolerance, rmg.relativeTolerance)

    #reset the reaction model to its original state:
    rmg.reactionModel.core.reactions = original_reactions

    logging.info('Conversion of reduced model ({} rxns): {:.2f}%'.format(no_important_reactions, conversion * 100))
    return conversion

def optimize_tolerance(target_label, reactionModel, rmg, reaction_system_index, error, orig_conv):
    """
    Increment the trial tolerance from a very low value until the introduced error is greater than the parameter threshold.
    """


    start = 1E-20
    incr = 10
    tolmax = 1

    tol = start
    trial = start
    while True:
        logging.info('Trial tolerance: {trial:.2E}'.format(**locals()))
        Xred = reduce_compute(trial, target_label, reactionModel, rmg, reaction_system_index)
        dev = np.abs((Xred - orig_conv) / orig_conv)
        logging.info('Deviation: {dev:.2f}'.format(**locals()))

        if dev > error or trial > tolmax:
            break

        tol = trial
        trial = trial * incr

    if tol == start:
        logging.error('Starting value for tolerance was too high...')

    return tol

class ConcentrationListener(object):
    """Returns the species concentration profiles at each time step."""

    def __init__(self):
        self.species_names = []
        self.data = []

    def update(self, subject):
        self.data.append((subject.t , subject.coreSpeciesConcentrations))


def main():
    
    # python -m scoop reduction.py models/minimal/input.py models/minimal/reduction.py models/minimal/chemkin/chem.inp
    inputFile, reductionFile, chemkinFile = sys.argv[-3:]

    for f in [inputFile, reductionFile, chemkinFile]:
        assert os.path.isfile(f)

    inputDirectory = os.path.abspath(os.path.dirname(inputFile))
    output_directory = inputDirectory

    rmg, target_label, error = load(inputFile, reductionFile, chemkinFile)

    reactionModel = rmg.reactionModel
    initialize(rmg.outputDirectory, reactionModel.core.reactions)

    atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
    index = 0
    reactionSystem = rmg.reactionSystems[index]

    print 'Allowed error in target conversion: {0:.0f}%'.format(error * 100)
    
    #compute original target conversion
    Xorig = compute_conversion(target_label, reactionModel, reactionSystem, index,\
     rmg.absoluteTolerance, rmg.relativeTolerance)
    print 'Original target conversion: {0:.0f}%'.format(Xorig * 100)

    # optimize reduction tolerance
    tol = optimize_tolerance(target_label, reactionModel, rmg, index, error, Xorig)
    print 'Optimized tolerance: {:.0E}'.format(tol)

def loadReductionInput(reductionFile):
    """
    Load an reduction job from the input file located at `reductionFile`
    """

    target = None
    tolerance = -1

    full_path = os.path.abspath(os.path.expandvars(reductionFile))
    try:
        f = open(full_path)
    except IOError, e:
        logging.error('The input file "{0}" could not be opened.'.format(full_path))
        logging.info('Check that the file exists and that you have read access.')
        raise e

    logging.info('Reading input file "{0}"...'.format(full_path))
    
    global_context = { '__builtins__': None }
    local_context = {
        '__builtins__': None,
        'target': target,
        'tolerance': tolerance
    }

    try:
        exec f in global_context, local_context

        target = local_context['target']
        tolerance = local_context['tolerance']

    except (NameError, TypeError, SyntaxError), e:
        logging.error('The input file "{0}" was invalid:'.format(full_path))
        logging.exception(e)
        raise
    finally:
        f.close()

    assert target is not None
    assert tolerance != -1

    return target, tolerance

def load(rmg_inputFile, reductionFile, chemkinFile):
    
    rmg = loadRMGPyJob(rmg_inputFile, chemkinFile)
    target, tolerance = loadReductionInput(reductionFile)

    return rmg, target, tolerance

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

if __name__ == '__main__':
    main()