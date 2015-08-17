import unittest

from reduction import *    

class MockMolecule(object):
    """docstring for MockMolecule"""
    def __init__(self, label):
        super(MockMolecule, self).__init__()
        self.label = label
        
class ReductionReactionTest(unittest.TestCase):

    def setUp(self):
        from rmgpy.reaction import Reaction

        mol1 = MockMolecule(label='mol1')
        mol2 = MockMolecule(label='mol2')
        mol3 = MockMolecule(label='mol3')
        mol4 = MockMolecule(label='mol4')
        
        self.rxn = Reaction(reactants=[mol1, mol2], products=[mol3, mol4])
        
        self.rrxn = ReductionReaction(self.rxn)


    def tearDown(self):
        del self.rrxn


    def test_constructor(self):
        rrxn = self.rrxn
        rxn = self.rxn

        self.assertIsNotNone(rrxn)

        # attributes
        self.assertIsNotNone(rrxn.reactants, rxn.reactants)
        self.assertIs(rrxn.products, rxn.products)
        self.assertIs(rrxn.rmg_reaction, rxn)
        self.assertIsNotNone(rrxn.stoichio)
        self.assertIsNone(rrxn.kf)
        self.assertIsNone(rrxn.kb)


        # stoichio
        for k,d in self.rrxn.stoichio.iteritems():
            for k,v in d.iteritems():
                self.assertEquals(v, 1)



    def test_reduce(self):
        import pickle
        reaction = pickle.loads(pickle.dumps(self.rrxn))


class OptimizeTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_optimize(self):
        #MINIMAL
        working_dir = 'models/minimal/'
        inputFile = working_dir+'input.py'
        chemkinFile = working_dir+'chemkin/chem.inp'
        reductionFile = working_dir+'reduction.py'

        rmg, target_label, error = load(inputFile, reductionFile, chemkinFile)

        reactionModel = rmg.reactionModel
        initialize(rmg.outputDirectory, reactionModel.core.reactions)

        atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
        index = 0
        reactionSystem = rmg.reactionSystems[index]
        
        # optimize reduction tolerance
        tol = optimize_tolerance(target_label, reactionModel, rmg, index, error, Xorig)
        print 'Optimized tolerance: {:.0E}'.format(tol)




if __name__ == '__main__':
    unittest.main()