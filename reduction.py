import itertools
import copy
import os.path
import csv
import numpy as np
from math import ceil

from rmgpy.chemkin import loadChemkinFile
from rmgpy.rmg.main import RMG
from rmgpy.solver.base import TerminationTime, TerminationConversion
from rmgpy.species import Species


"""
Guidelines for the chemkin input files:

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
    
    def __str__(self):
        return str(self.rmg_reaction)

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
    

class ReductionDriver(object):
    """docstring for ReductionDriver"""
    def __init__(self, core_species, working_dir, tolerance, number_of_time_steps=None):
        super(ReductionDriver, self).__init__()

        """
        A list with species objects.
        """
        self.core_species = core_species


        """
        Working dir is the directory in which we look for the csv file with the simulation profiles.

        """
        self.working_dir = working_dir

        """
        Tolerance to decide whether a reaction is unimportant for the formation/destruction of a species

        Tolerance is a floating point value between 0 and 1.

        A high tolerance means that many reactions will be deemed unimportant, and the reduced model will be drastically
        smaller.

        A low tolerance means that few reactions will be deemed unimportant, and the reduced model will only differ from the full
        model by a few reactions.
        """

        self.tolerance = tolerance


        """

        Number of time steps between start and end time of the batch reactor simulation at which the importance of 
        reactions should be evaluated.



        The more timesteps, the less chance we have to remove an important reactions, but the more simulations
        need to be carried out.
        """
        if number_of_time_steps is not None:
            self.number_of_time_steps = number_of_time_steps


def read_simulation_profile(filepath):
    """

    Reads in a csv file, located in the working directory under the name
    'simulation_1.csv'.

    This file contains the intermediate state variables
    (time, volume, core species concentrations) of the reactor simulation.

    Returns a table, with each row the core species concentrations
    (3rd to last columns) at the specified time (1st column). The
    computed volume is reported in column 2.

    First row of table are the headers.

    """


    with open(filepath, 'rb') as f:
        profile = list(csv.reader(f))

    raw = profile[1:]

    #massage data somewhat:
    processed = []
    for row in raw:
        processed.append(map(float,row))

    return np.array(processed)

def simulate_one(outputDirectory, reactionModel, atol, rtol, index, reactionSystem):
    """

    Simulates one reaction system, writes the results
    to a csv file 'simulation_X' with X the index.


    """
    csvfile = file(os.path.join(outputDirectory, 'simulation_{0}.csv'.format(index)),'w')
    worksheet = csv.writer(csvfile)

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
        worksheet = worksheet,
        absoluteTolerance = atol,
        relativeTolerance = rtol,
    ) 

    assert terminated

def simulate_all(rmg):
    """
    Simulate the RMG job, generating an output csv file
    for each of the simulated reaction systems.
    """
    
    reactionModel = rmg.reactionModel
    assert rmg.saveSimulationProfiles
    atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
    for index, reactionSystem in enumerate(rmg.reactionSystems):
        simulate_one(rmg.outputDirectory, reactionModel, atol, rtol, index, reactionSystem)
        


def initialize(core_species, working_dir, tol, number_of_time_steps):
    """
    Create a global reduction driver variable that will share its state
    with functions that need species concentrations, process conditions.

    Takes a reactionModel and a reactionSystem object as an argument.

    """
    global reduction
    assert os.path.isdir(working_dir)
    reduction = ReductionDriver(core_species, working_dir, tol, number_of_time_steps=number_of_time_steps)


def find_unimportant_reactions(reactions, rmg):
    """
    This function:

    - loops over all reactions
    - loops over all the species involved in a specific reaction
    - decides whether the specific reaction is important for the species.

    Whenever it is found that a reaction is important for a species, we break
    the species loop, and keep the reaction in the model.


    Returns:
        a list of reactions that can be removed.
    """
    # print 'the number of timesteps stored in the ReductionDriver: ', reduction.number_of_time_steps

    # run the simulation, creating csv concentration profiles for each reaction system defined in input.
    simulate_all(rmg)

    # start the model reduction
    reduce_reactions = [ReductionReaction(rxn) for rxn in reactions]

    closure = assess_reaction_closure(rmg.reactionSystems, reduce_reactions, reduction.tolerance)
    reactions_to_be_removed = list(itertools.ifilterfalse(closure, reduce_reactions))

    # reactions_to_be_removed = []
    # for rxn_j in model:
    #     isImportant = assess_reaction(rxn_j, profile, model, indices)
    #     if not isImportant: reactions_to_be_removed.append(rxn_j)


    return [rxn.rmg_reaction for rxn in reactions_to_be_removed]


def assess_reaction_closure(reactionSystems, reactions, tolerance):
    """
    Closure to be able to pass in the profile, reactions, and indices objects to the 
    assess_reaction function.
    """
    def myfilter(rxn_j):
        isImportant = assess_reaction(rxn_j, reactionSystems, reactions, tolerance)
        print "Is rxn {} important? {} ".format(rxn_j, isImportant)
        return isImportant
    
    return myfilter

def assess_reaction(rxn, reactionSystems, reactions, tolerance):
    """
    Returns whether the reaction is important or not in the reactions.

    It iterates over the reaction systems, and loads the concentration profile 
    of each reaction system into memory.

    It iterates over a number of samples in profile and 
    evaluates the importance of the reaction at every sample.


    """
    wd = reduction.working_dir

    # read in the intermediate state variables
    for system_index, reactionSystem in enumerate(reactionSystems):

        path = os.path.join(wd, 'simulation_{0}.csv'.format(system_index))
        assert os.path.isfile(path)
        profile = read_simulation_profile(path)
        T, P = reactionSystem.T.value_si, reactionSystem.P.value_si
        

        # take N evenly spaced indices from the table with simulation results:
        assert reduction.number_of_time_steps < len(profile)
        indices = np.linspace(0, len(profile)-1, num = reduction.number_of_time_steps)
        # samples = [profile[index][0] for index in indices]
        # print 'Time samples: ', samples

        for index in indices:
            assert profile[index] is not None

            timepoint = profile[index][0]# first column
            # print 'Timepoint: ', timepoint
            coreSpeciesConcentrations = profile[index][2:]# 3rd to last column
            # print coreSpeciesConcentrations
            
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
    

def get_stoichiometric_coefficient(rxn_j, spc_i, reactant_or_product):
    """
    ...
    """
    
    if reactant_or_product == 'reactant':
        stoich = 0
        for reactant in rxn_j.reactants:
            if reactant is spc_i: stoich -= 1
        return stoich
    elif reactant_or_product == 'product':
        stoich = 0
        for product in rxn_j.products:
            if product is spc_i: stoich += 1
        return stoich

    raise Exception('The species was not found in the reaction! Something went wrong!')   

def compute_reaction_rate(rxn_j, forward_or_reverse, T, P, coreSpeciesConcentrations): 
    """

    Computes reaction rate r as follows:

    r = k * Product(Ci^nu_ij, for all j)
    with:

    k = rate coefficient for rxn_j,
    Cij = the concentration for molecule i ,
    nu_ij = the stoichiometric coefficient for molecule i in reaction j.

    ...
    """

    if forward_or_reverse == 'forward':
        k = rxn_j.getRateCoefficient(T,P)
        species_list = rxn_j.reactants
        reactant_or_product = 'reactant'
    elif forward_or_reverse == 'reverse':
        kb = rxn_j.getReverseRateCoefficient(T,P)
        k = kb
        species_list = rxn_j.products
        reactant_or_product = 'product'

    assert species_list is not None

    concentrations = []
    for spc_i in species_list:
        ci = getConcentration(spc_i, coreSpeciesConcentrations)
        nu_i = get_stoichiometric_coefficient(rxn_j, spc_i, reactant_or_product)
        nu_i = abs(nu_i)
        #print 'The stoichiometric coefficient is: ',  nu_i
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
    assert isinstance(spc, Species)

    spc_index = -1

    for index, spc_core in enumerate(reduction.core_species):
        if spc is spc_core:#TODO reference comparison!
            spc_index = index
            break

    assert spc_index != -1
    return coreSpeciesConcentrations[spc_index]

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
   
    nu_i = get_stoichiometric_coefficient(rxn_j, spc_i, reactant_or_product)
    # print 'stoichio metric coeff of spc {} in rxn {}: {} '.format( spc_i, rxn_j, nu_i)
    if reactant_or_product == 'reactant':
        forward_or_reverse = 'forward'
    elif reactant_or_product == 'product':
        forward_or_reverse = 'reverse'

    r_j = compute_reaction_rate(rxn_j, forward_or_reverse, T, P, coreSpeciesConcentrations)

    rij = nu_i * r_j
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
        if spc_i in molecules:
            rij = calc_rij(reaction, spc_i,  reactant_or_product, T, P, coreSpeciesConcentrations)
            rate = rate + rij
        else:
            pass
            #print 'This species is not part of this reaction. Ignoring this reaction.'

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
    print 'Is reaction {0} important for species {1}: {2}'.format(rxn, spc, important)
 

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
