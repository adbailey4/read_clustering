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
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt
import tempfile
import numpy as np


class VariantCallTests(unittest.TestCase):
    """Test VariantCall class methods"""

    @classmethod
    def setUpClass(cls):
        super(VariantCallTests, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-2])
        cls.variant_call_file = os.path.join(cls.HOME, "tests/test_files/test_variant_call.csv")
        cls.vc = VariantCall(cls.variant_call_file)
        cls.pos = [27, 99, 105, 119, 210, 301, 413, 419, 435, 465, 540, 561, 577, 618, 631, 758, 765, 795, 973, 998,
                   1006, 1125, 1180, 1186,
                   1190, 1268, 1270, 1279, 1289, 1414, 1427, 1571, 1574, 1638, 1772, 1780, 1781]

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
        expected_1 = self.vc.data.iloc[
                     [87, 88, 89, 91, 95, 96, 97, 98, 103, 105, 106, 107, 108, 110, 111, 112, 114, 116, 117, 118, 122,
                      123,
                      124, 126, 127, 129, 130, 134], :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variant_data(id_1, 'T'))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[138, 143, 147, 148], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_data(id_2, 'c'))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[323, 324, 325, 326, 330, 335, 336, 337, 340, 343, 344, 349, 350], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_data(id_3, 'l'))

    def test_get_read_variant_set_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[92, 104, 132, 133], :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variant_set_data(id_1, 'Cb'))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[152, 153], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_set_data(id_2, 'Aj'))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[321, 322, 328, 329, 331, 334, 338, 339], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_set_data(id_3, 'Aa'))

    def test_get_read_variants_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[92, 99, 104, 125, 132, 133], :]
        pd.testing.assert_frame_equal(expected_1, self.vc.get_read_variants_data(id_1, ['C', 'e']))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[136, 138, 139, 140, 143, 145, 146, 147, 148, 149, 152, 153], :]
        pd.testing.assert_frame_equal(expected_2.reset_index(drop=True),
                                      self.vc.get_read_variants_data(id_2, ['G', 'k', 'l', 'j']).reset_index(drop=True))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[321, 322, 328, 329, 331, 334, 338, 339, 356, 357], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variants_data(id_3, ['A', 'j']))

    def test_get_read_variant_sets_data(self):
        id_1 = '03e6757b-31de-4b13-ab18-57f375404f28'
        expected_1 = self.vc.data.iloc[[87, 99, 111, 112, 116, 117, 125, 127], :]
        pd.testing.assert_frame_equal(expected_1.reset_index(drop=True),
                                      self.vc.get_read_variant_sets_data(id_1, ['Td', 'Ce']).reset_index(drop=True))

        id_2 = '043a9f51-5127-4a29-bdfd-5154cf3fa3a7'
        expected_2 = self.vc.data.iloc[[137, 144, 149, 150, 151, 152, 153], :]
        pd.testing.assert_frame_equal(expected_2, self.vc.get_read_variant_sets_data(id_2, ['Gk', 'Ci', 'Cb', 'Aj']))

        id_3 = '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'
        expected_3 = self.vc.data.iloc[[327, 341, 345, 354], :]
        pd.testing.assert_frame_equal(expected_3, self.vc.get_read_variant_sets_data(id_3, ['Tg', 'Cb']))

    def test_get_variant_sets(self):
        temp_df = pd.DataFrame({'variants': ['Aa', 'Tl', 'Cb', 'Gc', 'Tg', 'Td', 'Ci', 'Gk', 'Aj', 'Tdm', 'Th',
                                             'Ce', 'Af']})
        assert_frame_equal(temp_df.reset_index(drop=True), (self.vc.get_variant_sets()).reset_index(drop=True),
                           check_dtype=False)

    def test_get_number_of_variant_sets(self):
        self.assertEqual(13, self.vc.get_number_of_variant_sets())

    def test_get_variant_set_data(self):
        temp_df = pd.DataFrame(
            {'read_id': ['02636c05-538f-4647-b3ec-8e3a8c5eb10e', '01196b69-900b-4dc0-95a7-169cd79fae9b',
                         '03e6757b-31de-4b13-ab18-57f375404f28', '04c78365-fd9f-4391-8ddb-277620028285',
                         '02d4da5c-ec95-43ac-ac61-9f98ef4a4ca1'],
             'contig': ['RDN25-1', 'RDN25-1', 'RDN25-1',
                        'RDN25-1', 'RDN25-1'], 'reference_index': ['2346', '2346', '2346', '2346', '2346'],
             'strand': ['+', '+', '+', '+', '+'],
             'variants': ['Tdm', 'Tdm', 'Tdm', 'Tdm', 'Tdm'],
             'prob1': ['0.902366', '0.722098', '0.739977', '0.794725', '0.840949'],
             'prob2': ['0.048817',
                       '0.138951', '0.130012', '0.102638', '0.079525'],
             'prob3': ['0.048817',
                       '0.138951', '0.130012', '0.102638', '0.079525']})
        temp_df = temp_df.astype({"reference_index": int, "prob1": float, "prob2": float, "prob3": float})
        pd.testing.assert_frame_equal(temp_df.reset_index(drop=True),
                                      (self.vc.get_variant_set_data("Tdm")).reset_index(drop=True),
                                      check_exact=False, check_less_precise=4)

    def test_get_positions_of_variant_set(self):
        temp_df = pd.DataFrame(
            {'contig': ['RDN25-1'],
             'reference_index': ['2346'],
             'strand': ['+'], 'variants': ['Tdm']})

        temp_df = temp_df.astype({"reference_index": int})
        assert_frame_equal(temp_df.reset_index(drop=True),
                           (self.vc.get_positions_of_variant_set("Tdm")).reset_index(drop=True),
                           check_dtype=False)

    def test_get_variant_sets_data(self):
        temp_df = pd.DataFrame(
            {'read_id': ['028a34d4-2a7a-44e7-ab23-305915996ec8', '02636c05-538f-4647-b3ec-8e3a8c5eb10e',
                         '01196b69-900b-4dc0-95a7-169cd79fae9b', '03e6757b-31de-4b13-ab18-57f375404f28',
                         '043a9f51-5127-4a29-bdfd-5154cf3fa3a7', '031b1662-cbda-4efd-9120-257ac7b32eea',
                         '02c6037c-d73b-414d-9090-0bfe88a1e0b0', '04ac5ad4-d0d2-4bb4-bc8b-ff3b713661dc',
                         '02381d7b-ad58-4d21-8ee3-f77401c13814', '04c78365-fd9f-4391-8ddb-277620028285',
                         '02d2f886-87ff-4ab4-98f1-3eeb642f00c2', '02d4da5c-ec95-43ac-ac61-9f98ef4a4ca1', ],
             'contig': ['RDN18-1', 'RDN25-1', 'RDN25-1', 'RDN25-1', 'RDN18-1', 'RDN18-1', 'RDN18-1', 'RDN18-1',
                        'RDN18-1', 'RDN25-1', 'RDN18-1', 'RDN25-1'],
             'reference_index': ['1574', '2346', '2346', '2346', '1574', '1574', '1574', '1574', '1574', '2346', '1574',
                                 '2346'],
             'strand': ['+', '+', '+', '+', '+', '+', '+', '+', '+', '+', '+', '+'],
             'variants': ['Gk', 'Tdm', 'Tdm', 'Tdm', 'Gk', 'Gk', 'Gk', 'Gk', 'Gk', 'Tdm', 'Gk', 'Tdm'],
             'prob1': ['0.256589', '0.902366', '0.722098', '0.739977', '0.970518', '0.947118', '0.301744', '0.496635',
                       '0.215732', '0.794725', '1.000000', '0.840949'],
             'prob2': ['0.743411', '0.048817', '0.138951', '0.130012', '0.029482', '0.052882', '0.698256', '0.503365',
                       '0.784268', '0.102638', '0.000000', '0.079525'],
             'prob3': ['NaN', '0.048817', '0.138951', '0.130012', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', '0.102638',
                       'NaN', '0.079525']})

        temp_df = temp_df.astype({"reference_index": int, "prob1": float, "prob2": float, "prob3": float})
        pd.testing.assert_frame_equal(temp_df.reset_index(drop=True),
                                      (self.vc.get_variant_sets_data(['Tdm', 'Gk'])).reset_index(drop=True),
                                      check_exact=False, check_less_precise=4)

    def test_get_positions_of_variant_sets(self):
        temp_df = pd.DataFrame(
            {'contig': ['RDN18-1', 'RDN25-1'],
             'reference_index': ['1574', '2346'],
             'strand': ['+', '+'],
             'variants': ['Gk', 'Tdm']})
        temp_df = temp_df.astype({"reference_index": int})
        assert_frame_equal(temp_df.reset_index(drop=True),
                           (self.vc.get_positions_of_variant_sets(['Tdm', 'Gk'])).reset_index(drop=True),
                           check_dtype=False)

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

    def test_plot_number_reads_covering_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_file = os.path.join(temp_dir, "fake_file.png")
            out_file, n_reads = self.vc.plot_number_reads_covering_positions("RDN18-1", fake_file)
            self.assertEqual(out_file, fake_file)
            self.assertEqual(2, n_reads)

    def test_affinity_propagation(self):
        predict_test = np.asarray([0, 1])
        fit, predictor = self.vc.affinity_propagation(self.pos)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_gaussian_mixture_models(self):
        predict_test = np.asarray([1, 0])
        fit, predictor = self.vc.gaussian_mixture_models(self.pos, n_clusters=2)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_agglomerative_clustering(self):
        predict_test = np.asarray([1, 0])
        fit, predictor = self.vc.agglomerative_clustering(self.pos, n_clusters=2)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_DBSCAN(self):
        predict_test = np.asarray([0, 1])
        fit, predictor = self.vc.DBSCAN(self.pos, eps=0.5, min_samples=1)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_HDBSCAN(self):
        predict_test = np.asarray([-1, -1])
        fit, predictor = self.vc.HDBSCAN(self.pos, cluster_size=2)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_k_means(self):
        predict_test = np.asarray([1, 0])
        fit, predictor = self.vc.k_means(self.pos, max_number_clusters=2, find_optimal=False)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_mean_shift(self):
        predict_test = np.asarray([0, 1])
        fit, predictor = self.vc.mean_shift(self.pos, find_optimal=False)
        self.assertEqual(predict_test.all(), predictor.all())

    def test_spectral_clustering(self):
        predict_test = np.asarray([1, 0])
        fit, predictor = self.vc.spectral_clustering(self.pos, n_clusters=2, affinity='rbf')
        self.assertEqual(predict_test.all(), predictor.all())

    def test_plot_tSNE_reads_covering_positions_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_file = os.path.join(temp_dir, "fake_file.png")
            fig_path = self.vc.plot_tSNE_reads_covering_positions_data(self.pos, 'HDBSCAN', fake_file, cluster_size=2)
            self.assertEqual(fig_path, fake_file)

    def test_plot_PCA_reads_covering_positions_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_file = os.path.join(temp_dir, "fake_file.png")
            fig_path = self.vc.plot_tSNE_reads_covering_positions_data(self.pos, 'HDBSCAN', fake_file, cluster_size=2)
            self.assertEqual(fig_path, fake_file)

    def test_get_dendrogram(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_file = os.path.join(temp_dir, "fake_file.png")
            list_18 = self.vc.get_contig_positions('RDN18-1')
            fig_path = self.vc.get_dendrogram(list_18, y1=1.35, y2=1, y3=0.9, figure_path=fake_file)
            self.assertEqual(fig_path, fake_file)

    def test_get_reads_covering_positions_data1(self):
        positions = [435, 465]
        first_col = ''
        sec_col = ''
        first_col += 'P' + ' ' + str(positions[0])
        sec_col += 'P' + ' ' + str(positions[1])

        temp_df = pd.DataFrame(
            {'read_id': ['02381d7b-ad58-4d21-8ee3-f77401c13814', '02c6037c-d73b-414d-9090-0bfe88a1e0b0',
                         '02d2f886-87ff-4ab4-98f1-3eeb642f00c2', '031b1662-cbda-4efd-9120-257ac7b32eea',
                         '04ac5ad4-d0d2-4bb4-bc8b-ff3b713661dc'],
             first_col: ['0.208556', '0.809814', '0.038271', '0.019851', '1.000000'],
             sec_col: ['0.989057', '1.000000', '0.000000', '0.643448', '0.070223']})

        temp_df = temp_df.astype({first_col: float, sec_col: float})
        pd.testing.assert_frame_equal(temp_df.reset_index(drop=True),
                                      (self.vc.get_reads_covering_positions_data(positions)).reset_index(
                                          drop=True), check_exact=False, check_less_precise=4)

    def test_get_reads_covering_positions_data2(self):
        positions = [435, 465, 561]
        first_col = ''
        sec_col = ''
        third_col = ''
        first_col += 'P' + ' ' + str(positions[0])
        sec_col += 'P' + ' ' + str(positions[1])
        third_col += 'P' + ' ' + str(positions[2])

        temp_df = pd.DataFrame(
            {'read_id': ['02381d7b-ad58-4d21-8ee3-f77401c13814', '02c6037c-d73b-414d-9090-0bfe88a1e0b0',
                         '02d2f886-87ff-4ab4-98f1-3eeb642f00c2', '031b1662-cbda-4efd-9120-257ac7b32eea',
                         '04ac5ad4-d0d2-4bb4-bc8b-ff3b713661dc'],
             first_col: ['0.208556', '0.809814', '0.038271', '0.019851', '1.000000'],
             sec_col: ['0.989057', '1.000000', '0.000000', '0.643448', '0.070223'],
             third_col: ['0.181277', '0.000000', '0.070602', '0.818655', '0.538190']})

        temp_df = temp_df.astype({first_col: float, sec_col: float, third_col: float})

        pd.testing.assert_frame_equal(temp_df.reset_index(drop=True),
                                      (self.vc.get_reads_covering_positions_data(positions)).reset_index(
                                          drop=True), check_exact=False, check_less_precise=4)

    def test_get_reads_covering_positions_data3(self):
        positions = [435, 27]
        first_col = ''
        sec_col = ''
        first_col += 'P' + ' ' + str(positions[0])
        sec_col += 'P' + ' ' + str(positions[1])

        temp_df = pd.DataFrame(
            {'read_id': ['02381d7b-ad58-4d21-8ee3-f77401c13814', '02c6037c-d73b-414d-9090-0bfe88a1e0b0',
                         '02d2f886-87ff-4ab4-98f1-3eeb642f00c2'],
             first_col: ['0.208556', '0.809814', '0.038271'],
             sec_col: ['0.0', '0.0', '0.0']})

        temp_df = temp_df.astype({first_col: float, sec_col: float})

        pd.testing.assert_frame_equal(temp_df.reset_index(drop=True),
                                      (self.vc.get_reads_covering_positions_data(positions)).reset_index(
                                          drop=True), check_exact=False, check_less_precise=4)


if __name__ == '__main__':
    unittest.main()