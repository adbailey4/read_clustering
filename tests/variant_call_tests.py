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

    def test_get_number_of_positions(self):
        contig = "RDN18-1"
        strand = "+"
        self.assertEqual(37, self.vc.get_number_of_positions(contig, strand))
        contig = "RDN18-1"
        strand = "-"
        self.assertEqual(0, self.vc.get_number_of_positions(contig, strand))
        contig = "RDN25-1"
        strand = "-"
        self.assertEqual(0, self.vc.get_number_of_positions(contig, strand))
        contig = "RDN25-1"
        strand = "+"
        self.assertEqual(48, self.vc.get_number_of_positions(contig, strand))

    def test_get_position_data(self):
        contig = "RDN18-1"
        strand = "+"
        position = 1
        self.assertFalse(self.vc.get_position_data(contig, strand, position))
        contig = "RDN18-1"
        strand = "+"
        position = 973
        self.assertEqual(7, len(self.vc.get_position_data(contig, strand, position)))
        self.assertSetEqual({position}, set(self.vc.get_position_data(contig, strand, position)["reference_index"]))
        self.assertSetEqual({strand}, set(self.vc.get_position_data(contig, strand, position)["strand"]))
        self.assertSetEqual({contig}, set(self.vc.get_position_data(contig, strand, position)["contig"]))

    def test_get_positions_data(self):
        contigs = ["RDN18-1"]
        strands = ["+"]
        positions = [1]
        self.assertFalse(self.vc.get_positions_data(contigs, strands, positions))
        contigs = ["RDN18-1", "RDN18-1"]
        strands = ["+", "+"]
        positions = [973, 1268]
        self.assertEqual(14, len(self.vc.get_positions_data(contigs, strands, positions)))
        self.assertSetEqual(set(positions),
                            set(self.vc.get_positions_data(contigs, strands, positions)["reference_index"]))
        self.assertSetEqual(set(strands),
                            set(self.vc.get_positions_data(contigs, strands, positions)["strand"]))
        self.assertSetEqual(set(contigs),
                            set(self.vc.get_positions_data(contigs, strands, positions)["contig"]))

    def test_get_variant_sets_from_position(self):
        contig = "RDN18-1"
        strand = "+"
        position = 1
        self.assertFalse(self.vc.get_variant_sets_from_position(contig, strand, position))
        contig = "RDN18-1"
        strand = "+"
        position = 973
        self.assertSetEqual({"Aa"}, self.vc.get_variant_sets_from_position(contig, strand, position))

    def test_get_variants_from_position(self):
        contig = "RDN18-1"
        strand = "+"
        position = 1
        self.assertFalse(self.vc.get_variants_from_position(contig, strand, position))
        contig = "RDN18-1"
        strand = "+"
        position = 973
        self.assertSetEqual({"A", "a"}, self.vc.get_variants_from_position(contig, strand, position))

    def test_get_read_position_data(self):
        read_id = "043a9f51-5127-4a29-bdfd-5154cf3fa3a7"
        position = 1
        self.assertFalse(self.vc.get_read_position_data(read_id, position))
        read_id = "043a9f51-5127-4a29-bdfd-5154cf3fa3a7"
        position = 973
        data = self.vc.get_read_position_data(read_id, position)
        self.assertEqual(1, len(data))
        self.assertEqual(position, data["reference_index"].iloc[0])
        self.assertEqual(read_id, data["read_id"].iloc[0])

    def test_get_read_positions_data(self):
        read_id = "043a9f51-5127-4a29-bdfd-5154cf3fa3a7"
        positions = [1, 0]
        self.assertFalse(self.vc.get_read_positions_data(read_id, positions))
        read_id = "043a9f51-5127-4a29-bdfd-5154cf3fa3a7"
        positions = [973, 1]
        data = self.vc.get_read_positions_data(read_id, positions)
        self.assertEqual(1, len(data))
        self.assertEqual(973, data["reference_index"].iloc[0])
        self.assertEqual(read_id, data["read_id"].iloc[0])
        positions = [973, 1268]
        data = self.vc.get_read_positions_data(read_id, positions)
        self.assertEqual(2, len(data))
        self.assertSetEqual({973, 1268}, set(data["reference_index"]))
        self.assertSetEqual({read_id}, set(data["read_id"]))


if __name__ == '__main__':
    unittest.main()
