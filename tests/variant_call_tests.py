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
        cls.variant_call_file = os.path.join(cls.HOME, "tests/test_files/test_variant_call.csv")
        cls.vc = VariantCall(cls.variant_call_file)

    def test_variant_call_file(self):
        self.assertTrue(os.path.exists(self.variant_call_file))

    def test_VariantCall(self):
        self.assertIsInstance(self.vc, VariantCall)

    def test_get_read_ids(self):
        self.assertSetEqual({
            '03e6757b-31de-4b13-ab18-57f375404f28',
            '02381d7b-ad58-4d21-8ee3-f77401c13814',
            '02d2f886-87ff-4ab4-98f1-3eeb642f00c2',
            '02c6037c-d73b-414d-9090-0bfe88a1e0b0',
            '031b1662-cbda-4efd-9120-257ac7b32eea',
            '04ac5ad4-d0d2-4bb4-bc8b-ff3b713661dc',
            '043a9f51-5127-4a29-bdfd-5154cf3fa3a7',
            '02636c05-538f-4647-b3ec-8e3a8c5eb10e',
            '04c78365-fd9f-4391-8ddb-277620028285',
            '02d4da5c-ec95-43ac-ac61-9f98ef4a4ca1',
            '028a34d4-2a7a-44e7-ab23-305915996ec8',
            '01196b69-900b-4dc0-95a7-169cd79fae9b'
        }, self.vc.get_read_ids())

    def test_get_number_of_ids(self):
        self.assertEqual(12, self.vc.get_number_of_ids())

    def test_get_strands(self):
        self.assertSetEqual({"+"}, self.vc.get_strands())

    def test_get_number_of_strands(self):
        self.assertEqual(1, self.vc.get_number_of_strands())


if __name__ == '__main__':
    unittest.main()
