import os
import sys
import unittest

from scoop import futures, _control, shared
from scoop import logger as logging

from reduction import compute_conversion, initialize, load, optimize_tolerance, reduce_compute
from scoop_framework.framework import TestScoopCommon


class OptimizeTest(TestScoopCommon):

    #MINIMAL
    working_dir = 'models/minimal/'
    inputFile = working_dir+'input.py'
    chemkinFile = working_dir+'chemkin/chem.inp'
    reductionFile = working_dir+'reduction_input.py'

    def __init__(self, *args, **kwargs):
        # Parent initialization
        super(self.__class__, self).__init__(*args, **kwargs)
        
        # Only setup the scoop framework once, and not in every test method:
        super(self.__class__, self).setUp()

    @classmethod
    def setUpClass(cls):
        super(OptimizeTest, cls).setUpClass()
        rmg, target_label, error = load(cls.inputFile, cls.reductionFile, cls.chemkinFile)
        cls.rmg = rmg
        cls.target_label = target_label
        cls.error = error

        reactionModel = rmg.reactionModel
        result = futures._startup(initialize, rmg.outputDirectory, reactionModel.core.reactions)
    

    def test_compute_conversion(self):
        rmg = OptimizeTest.rmg
        target_label = OptimizeTest.target_label
        reactionModel = rmg.reactionModel

        atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
        index = 0
        reactionSystem = rmg.reactionSystems[index]

        conv = compute_conversion(target_label, reactionModel, reactionSystem, index,\
         rmg.absoluteTolerance, rmg.relativeTolerance)
        self.assertIsNotNone(conv)


    def test_reduce_compute(self):
        rmg = OptimizeTest.rmg
        target_label = OptimizeTest.target_label
        reactionModel = rmg.reactionModel


        atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
        index = 0
        reactionSystem = rmg.reactionSystems[index]

        orig_conv = compute_conversion(target_label, reactionModel, reactionSystem, index,\
         rmg.absoluteTolerance, rmg.relativeTolerance)

        tols = [0.7, 1e-3, 1e-6]
        for tol in tols:
            conv = reduce_compute(tol, target_label, reactionModel, rmg, index)
            self.assertIsNotNone(conv)

    def test_optimize(self):
        rmg = OptimizeTest.rmg
        target_label = OptimizeTest.target_label
        error = OptimizeTest.error
        reactionModel = rmg.reactionModel


        atol, rtol = rmg.absoluteTolerance, rmg.relativeTolerance
        index = 0
        reactionSystem = rmg.reactionSystems[index]
        
        #compute original target conversion
        Xorig = compute_conversion(target_label, reactionModel, reactionSystem, index,\
         rmg.absoluteTolerance, rmg.relativeTolerance)

        # optimize reduction tolerance
        tol = optimize_tolerance(target_label, reactionModel, rmg, index, error, Xorig)
        self.assertAlmostEqual(1e-04, tol)


if __name__ == '__main__' and os.environ.get('IS_ORIGIN', "1") == "1":
    logging.setLevel(20)
    unittest.main()