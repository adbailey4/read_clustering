#!/usr/bin/env python
"""Tests for variant call class handler"""
########################################################################
# File: variant_call_tests.py
#
# Author: Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar
# History: 06/23/20 Created
########################################################################

import pandas as pd
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

    def test_get_read_data(self):
        id_1 = '028a34d4-2a7a-44e7-ab23-305915996ec8'
        expected_1 = self.vc.data.iloc[:19, :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_data(id_1))

        id_2 = '02636c05-538f-4647-b3ec-8e3a8c5eb10e'
        expected_2 = self.vc.data.iloc[19:51, :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_data(id_2))

        id_3 = '01196b69-900b-4dc0-95a7-169cd79fae9b'
        expected_3 = self.vc.data.iloc[51:87, :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_data(id_3))

        id_4 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_4 = self.vc.data.iloc[87:135, :]
        pd.testing.assert_frame_equal(expected_4, self.vc.get_read_data(id_4))

    def test_get_read_positions(self):
        id_1 = '028a34d4-2a7a-44e7-ab23-305915996ec8'
        expected = self.vc.data.loc[:, ['contig', 'strand', 'reference_index', 'variants']]
        expected_1 = expected.iloc[:19, :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_positions(id_1))

        id_2 = '02636c05-538f-4647-b3ec-8e3a8c5eb10e'
        expected_2 = expected.iloc[19:51, :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_positions(id_2))

        id_3 = '01196b69-900b-4dc0-95a7-169cd79fae9b'
        expected_3 = expected.iloc[51:87, :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_positions(id_3))

        id_4 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_4 = expected.iloc[87:135, :]
        pd.testing.assert_frame_equal(expected_4, self.vc.get_read_positions(id_4))

    def test_get_read_variant_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[87,88,89,91,95,96,97,98,103,105,106,107,108,110,111,112,114,116,117,118,122,123,
                                        124,126,127,129,130,134],:]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variant_data(id_1,'T'))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[138,143,147,148], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_data(id_2, 'c'))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[323,324,325,326,330,335,336,337,340,343,344,349,350], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_data(id_3, 'l'))

    def test_get_read_variant_set_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[92,104,132,133], :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variant_set_data(id_1, 'Cb'))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[152,153], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_set_data(id_2, 'Aj'))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[321,322,328,329,331,334,338,339], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_set_data(id_3, 'Aa'))


    def test_get_read_variants_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[92,99,104,125,132,133], :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variants_data(id_1, ['C','e']))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[136,138,139,140,143,145,146,147,148,149,152,153], :]
        pd.testing.assert_frame_equal(expected_2.reset_index(drop=True),
                                      self.vc.get_read_variants_data(id_2, ['G','k','l','j']).reset_index(drop=True))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[321, 322, 328, 329, 331, 334, 338, 339,356,357], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variants_data(id_3, ['A','j']))


    def test_get_read_variant_sets_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[87,99,111,112,116,117,125,127], :]
        pd.testing.assert_frame_equal(expected_1.reset_index(drop=True),
                                      self.vc.get_read_variant_sets_data(id_1, ['Td', 'Ce']).reset_index(drop=True))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[137,144,149,150,151,152,153], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_sets_data(id_2, ['Gk', 'Ci','Cb','Aj']))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[327,341,345,354], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_sets_data(id_3, ['Tg', 'Cb']))

if __name__ == '__main__':
    unittest.main()
