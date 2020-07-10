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

    def get_variant_sets(self):
        """Return the set of all possible sets of variants"""
        temp_df = self.data[['variants']]
        temp_df = temp_df.drop_duplicates()
        return temp_df

    def get_number_of_variant_sets(self):
        """Return the number of possible sets of variants"""
        return len(self.get_variant_sets())

    def get_variant_set_data(self, variant_set):
        """Return the corresponding data with specific variant set
        :param variant_set: a single variant set from the data
        :return temp_df: a data frame containing the data from the passed variant set
        """
        temp_df = self.data.loc[self.data['variants'] == variant_set]

        return temp_df

    def get_positions_of_variant_set(self, variant_set):
        """Return the contig, strand and position of all locations of a variant set
        :param variant_set: a single variant set from the data
        :return temp_df: a data frame containing the 'contig', 'reference_index', and 'strand' of a given variant
        """
        temp_df = self.data[self.data['variants'] == variant_set]
        temp_df = temp_df[['contig', 'reference_index', 'strand', 'variants']]
        temp_df = temp_df.drop_duplicates()

        return temp_df

    def get_variant_sets_data(self, variant_sets):
        """Return the corresponding data with a list of variant sets
        :param variant_sets: a list of numerous variant sets from the data
        :return temp_df: a data frame containing the data of all the variant_sets in the list
        """

        temp_df = self.data[self.data['variants'].isin(variant_sets)]
        temp_df = temp_df.drop_duplicates()

        return temp_df

    def get_positions_of_variant_sets(self, variant_sets):
        """Return the contig, strand and position of all locations of a list of variant sets
        :param variant_sets: a list of numerous variant sets from the data
        :return temp_df: a data frame containing the 'contig', 'reference_index', and 'strand' of all the variant_sets
                          in the list
        """

        final_df = self.data[self.data['variants'].isin(variant_sets)]
        final_df = final_df[['contig', 'reference_index', 'strand', 'variants']]

        final_df = final_df.drop_duplicates()

        return final_df

    def get_read_data(self, read_id):
        """Return the corresponding data with specific read_id.
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :return: data frame with all the corresponding data for the given read_id
        """
        df1 = self.data[self.data['read_id'] == read_id]
        return df1

    def get_read_positions(self, read_id):
        """Return the contig, strand and position of all locations covered by read
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :return: data frame with the 'contig', 'strand', 'reference_index', and 'variants' information
        for the given read_id
        """
        df1 = self.data[self.data['read_id'] == read_id]
        return df1.loc[:, ['contig','strand','reference_index','variants']]

    def get_read_variant_data(self, read_id, variant):
        """Return the corresponding data with specific read_id and specific variant
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :param variant: a single variant possibility for one nucleotide position. Python type: str
        :return: data frame with all the corresponding data for the given read_id and variant
        """
        df1 = self.get_read_data(read_id)
        return df1[df1.variants.str.contains(variant)]

    def get_read_variant_set_data(self, read_id, variant_set):
        """Return the corresponding data with specific read_id and specific variant set
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :param variant_set: all variant possibilities (2 or 3) for one nucleotide position. Python type:str
        :return: data frame with all the corresponding data for the given read_id and variant_set
        """
        df1 = self.get_read_data(read_id)
        return df1[df1['variants'] == variant_set]

    def get_read_variants_data(self, read_id, variants):
        """Return the corresponding data with specific read_id and list of variants
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :param variants: a list of single variant possibilities each for one different nucleotide position
        :return: data frame with all the corresponding data for the given read_id and variants
        """
        df1 = self.get_read_data(read_id)
        return df1[df1.variants.str.contains('|'.join(variants))]

    def get_read_variant_sets_data(self, read_id, variant_sets):
        """Return the corresponding data with specific read_id and list of variant sets
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :param variant_sets:  a list of all variant possibilities for different nucleotide positions
        :return: data frame with all the corresponding data for the given read_id and variant_sets
        """
        df1 = self.get_read_data(read_id)
        return df1[df1['variants'].isin(variant_sets)]

    def find_duplicates(self, read_id):
        """"return any duplicated rows if present in data set of a given read_id
        :param read_id: identification code for one continuous nucleotide reading. Python type:str
        :return: True if duplicated rows are present, False if not
        """
        df1 = self.get_read_data(read_id)
        dup = df1[df1.duplicated()]
        if dup.empty:
            return False
        else:
            return True

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

    def get_number_of_positions(self, contig, strand):
        """Return the number of variant positions on the contig strand
        :param contig: string of contig
        :param strand: string of strand {"+","-"}
        :return: integer of number of positions

        """
        new_df = self.data[(self.data['contig'] == contig) & (self.data['strand'] == strand)]
        return len(set(new_df.reference_index))

    def get_position_data(self, contig, strand, position):
        """If position exists return all data covering that position otherwise return false
        :param contig: string of contig
        :param strand: string of strand {"+","-"}
        :param position: integer of reference position
        :return dataframe where contig, strand and reference_index match input parameters
        """
        data = self.data[(self.data['contig'] == contig) &
                         (self.data['strand'] == strand) &
                         (self.data['reference_index'] == position)]
        if data.empty:
            return False
        else:
            return data

    def get_positions_data(self, contigs, strands, positions):
        """If positions exists return all data covering those positions otherwise return false

        ex: get_positions_data(["a", "b"], ["+", "+"], [10, 120])
        :param contigs: list of strings of contigs
        :param strands: list of strings of strands {"+","-"}
        :param positions: list of integer of reference positions
        :return dataframe where contig, strand and reference_index match input parameters
        """
        assert len(contigs) == len(strands) == len(positions), "Must pass in equal number of contigs, strands, " \
                                                               "and positions. "

        return self._get_positions_data(pd.DataFrame({"contig": contigs,
                                                      "strand": strands,
                                                      "reference_index": positions}))

    def _get_positions_data(self, unique_df):
        """If positions exists return all data covering those positions otherwise return false

        ex: get_positions_data(pd.DataFrame({"contig": contigs, "strand": strands, "reference_index": positions}))
        :param unique_df: DataFrame with "contig", "strand", "reference_index" columns and no duplicate rows
        :return dataframe where contig, strand and reference_index are in the input dataframe
        """
        assert sum(unique_df.duplicated()) == 0, "There are duplicated rows in the passed in data frame"
        keys = list(unique_df.columns.values)
        i1 = unique_df.set_index(keys).index
        i2 = self.data.set_index(keys).index
        data = self.data[i2.isin(i1)]
        if data.empty:
            return False
        else:
            return data

    def get_variant_sets_from_position(self, contig, strand, position):
        """If position exists in read return variant sets
        :param contig: string of contig
        :param strand: string of strand {"+","-"}
        :param position: integer of reference position
        :return: set of all variant sets or False if position does not exist
        """
        data = self.get_position_data(contig, strand, position)
        if data is not False:
            return set(data["variants"])
        else:
            return data

    def get_variants_from_position(self, contig, strand, position):
        """If position exists return variants
        :param contig: string of contig
        :param strand: string of strand {"+","-"}
        :param position: integer of reference position
        :return: set of all variants or False if position does not exist
        """
        data = self.get_variant_sets_from_position(contig, strand, position)
        if data is not False:
            return set("".join(data))
        else:
            return False

    def get_read_position_data(self, read_id, position):
        """If position exists in read return data covering position
        :param read_id: string read name
        :param position: integer reference index
        :return: dataframe with both read_id equal to read_id and reference_index equal to position
        """
        data = self.data[(self.data['read_id'] == read_id) & (self.data['reference_index'] == position)]
        if data.empty:
            return False
        assert len(data) == 1, \
            "Check input file. Got multiple hits for read {} and position {}.".format(read_id, position)
        return data

    def get_read_positions_data(self, read_id, positions):
        """If position exists in read return data covering list of positions
        :param read_id: string read name
        :param positions: list of reference indices
        :return: dataframe with both read_id equal to read_id and reference_index is in positions list
        """
        data = self.data[(self.data['read_id'] == read_id) & (self.data['reference_index'].isin(positions))]
        if data.empty:
            return False
        return data
    
    def get_reads_covering_2_positions_data(positions, variant_sets):
        """Select for the reads that cover both of the given positions and variant_sets, and return the corresponding prob2
        :param positions: 2 target positions (integers) given as a list
        :param variant_sets: 2 target variant_sets given as a list
        :return dataframe with 'read id' and the variant probabilities corresponding to the 2 positions given"""
        
        temp_df = data[(data['reference_index'].isin (positions)) & (data['variants'].isin (variant_sets))]
        plot_data = temp_df.loc[:,['read_id', 'reference_index', 'variants', 'prob1', 'prob2']]
        pos_n = len(positions)
        var_n = len(variant_sets)
        select = plot_data.groupby(['read_id']).nunique()     
        a = select[(select['reference_index'] == pos_n) & (select['variants'] == var_n)]
        a.columns = ['id_number', 'reference_index', 'variants', 'prob1', 'prob2']
        target_ids = list(a.index.values)        
        first_prob = []
        sec_prob = []
        for i in target_ids:
            r = (plot_data.loc[plot_data['read_id'] == i]).reset_index(drop=True)    
            first_prob.append(r.loc[0, 'prob2'])
            sec_prob.append(r.loc[1, 'prob2'])
        df_plot = pd.DataFrame(list(zip(target_ids, first_prob, sec_prob)))
        x_val = ''
        y_val = ''
        x_val += 'P' + ' ' + str(positions[0]) + ' ' + str(variant_sets[0])
        y_val += 'P' + ' ' + str(positions[1]) + ' ' + str(variant_sets[1])
        df_plot.columns = ['read id', x_val, y_val]             
    
        return df_plot
    
    def get_reads_covering_positions_data(self, positions, variant_sets):
        """Select for the reads that cover all given positions and variant_sets, and return the corresponding prob2
        :param positions: target positions (integers) given as a list
        :param variant_sets: target variant_sets given as a list
        :return dataframe with 'read id' and the variant probabilities corresponding to the positions given"""
        """return dataframe with probabilities for the modified variants in the given variant_set for the given position"""
        
        data = self.data[(self.data['reference_index'].isin (positions)) & (self.data['variants'].isin (variant_sets))]
        plot_data = data.loc[:,['read_id', 'reference_index', 'variants', 'prob1', 'prob2']]
        pos_n = len(positions)
        var_n = len(Counter(variant_sets).values())           
        select = plot_data.groupby(['read_id']).nunique()      
        
        a = select[(select['reference_index'] == pos_n) & (select['variants'] == var_n)]
        a.columns = ['id_number', 'reference_index', 'variants', 'prob1', 'prob2']
        target_ids = list(a.index.values)        
        
        d = {}
        for pos in positions:
            d[str(pos)] =[]      
        for i in target_ids:
            r = (plot_data.loc[plot_data['read_id'] == i]).set_index('reference_index')   
            for index, row in r.iterrows():          
                d[str(index)].append(r.at[index, 'prob2'])
        
        df_plot = pd.DataFrame(list(zip(target_ids)))
        df_plot.columns = ['Read ID']
        
        for key in d:
            col_val = ''
            col_val += 'P' + ' ' + str(key)
            df_plot[col_val] = d[key]   
      
        return df_plot
