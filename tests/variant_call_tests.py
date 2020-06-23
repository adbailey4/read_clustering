#!/usr/bin/env python
"""Tests for variant call class handler"""
########################################################################
# File: variant_call_tests.py
#
# Author: Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar
# History: 06/23/20 Created
########################################################################

import unittest
import os
from read_clustering.variant_call import VariantCall


class VariantCallTests(unittest.TestCase):
    """Test VariantCall class methods"""

    @classmethod
    def setUpClass(cls):
        super(VariantCallTests, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-2])
        cls.ivt_variant_call_file = os.path.join(cls.HOME, "tests/test_files/ivt.csv")

    def test_variant_call_file(self):
        self.assertTrue(os.path.exists(self.ivt_variant_call_file))

    def test_VariantCall(self):
        vc = VariantCall()
        self.assertIsInstance(vc, VariantCall)


if __name__ == '__main__':
    unittest.main()
