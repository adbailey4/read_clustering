#!/usr/bin/env python
"""Variant call file class handler"""
########################################################################
# File: variant_call.py
#
#
# Author: Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar
# History: 06/23/20 Created
########################################################################

import pandas as pd


class VariantCall(object):
    """Read in variant call file and give access points to various types of data"""

    def __init__(self, file_path):
        """
        Initialize the class
        :param file_path: path to csv file
        """
        self.data = pd.read_csv(file_path)
        self.canonical = {"A", "C", "G", "T"}

    def get_read_ids(self):
        """Return the set of read ids """
        return set(self.data.read_id)

    def get_number_of_ids(self):
        """Return the number of read ids """
        return len(self.get_read_ids())

    def get_strands(self):
        """Return the set of strands in the dataset"""
        return set(self.data.strand)

    def get_number_of_strands(self):
        """Return the number strands in the dataset"""
        return len(self.get_strands())

    ############################# VARIANT Set  #############################

    def get_variant_sets(self):
        """Return the set of all possible sets of variants"""
        pass

    def get_number_of_variant_sets(self):
        """Return the number of possible sets of variants"""
        pass

    def get_variant_set_data(self, variant_set):
        """Return the corresponding data with specific variant set"""
        pass

    def get_positions_of_variant_set(self, variant_set):
        """Return the contig, strand and position of all locations of a variant set"""
        pass

    def get_variant_sets_data(self, variant_sets):
        """Return the corresponding data with list of variant sets"""
        pass

    def get_positions_of_variant_sets(self, variant_sets):
        """Return the contig, strand and position of all locations of a list of variant sets"""
        pass

    ############################# READ ID #############################

    def get_read_data(self, read_id):
        """Return the corresponding data with specific read_id"""
        pass

    def get_read_positions(self, read_id):
        """Return the contig, strand and position of all locations covered by read"""
        pass

    def get_read_variant_data(self, read_id, variant):
        """Return the corresponding data with specific read_id and specific variant"""
        pass

    def get_read_variant_set_data(self, read_id, variant_set):
        """Return the corresponding data with specific read_id and specific variant set"""
        pass

    def get_read_variants_data(self, read_id, variants):
        """Return the corresponding data with specific read_id and list of variants"""
        pass

    def get_read_variant_sets_data(self, read_id, variant_sets):
        """Return the corresponding data with specific read_id and list of variant sets"""
        pass

    ############################# variant data #############################

    def get_number_of_variants(self):
        """Return the number of variants including canonical nucleotides"""
        pass

    def get_number_of_non_canonical_variants(self):
        """Return the number of variants NOT including canonical nucleotides"""
        pass

    def get_variant_data(self, variant):
        """Return the corresponding data with specific variant"""
        pass

    def get_variants_data(self, variants):
        """Return the corresponding data with list of variants"""
        pass

    def get_positions_of_variant(self, variant):
        """Return the contig, strand and position of all locations of a variant"""
        pass

    ############################# position data #############################

    def get_number_of_positions(self, contig, strand):
        """Return the number of variant positions on the contig strand"""
        new_df = self.data[(self.data['contig'] == contig) & (self.data['strand'] == strand)]
        return len(set(new_df.reference_index))

    def get_position_data(self, contig, strand, position):
        """If position exists return all data covering that position otherwise return false"""
        pass

    def get_positions_data(self, contigs, strands, positions):
        """If positions exists return all data covering those positions otherwise return false

        ex: get_positions_data(["a", "b"], ["+", "+"], [10, 120])
        """
        pass

    def get_data_from_position(self, contig, strand, position):
        """If position exists in read return data covering position"""
        pass

    def get_variants_from_position(self, contig, strand, position):
        """If position exists return variants"""
        pass

    def get_variant_sets_from_position(self, contig, strand, position):
        """If position exists in read return variant sets"""
        pass

    def get_read_position_data(self, read_id, position):
        """If position exists in read return data covering position"""
        pass

    def get_read_positions_data(self, read_id, positions):
        """If position exists in read return data covering list of positions"""
        pass

