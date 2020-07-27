#!/usr/bin/env python
"""Variant call file class handler"""
########################################################################
# File: variant_call.py
#
#
# Author: Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar
# History: 06/23/20 Created
########################################################################

import os
import math
import scipy
import hdbscan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from collections import Counter
from collections import defaultdict
from itertools import cycle
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer  # needs installation
from sklearn.metrics import silhouette_score
import shapely.geometry as SG  # INSTALL
from kneed import KneeLocator  # INSTALL


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
        return df1.loc[:, ['contig', 'strand', 'reference_index', 'variants']]

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

    def get_subunit_data(self, subunit):  # assumes there is at least one read that covers all positions
        data = self.data.loc[self.data['contig'] == 'RDN' + str(subunit[0:2]) + '-1']
        data_2 = data.groupby(['reference_index']).nunique()
        positions = []
        variants = []
        for i, row in data_2.iterrows():
            r = positions.append(i)
        return r

    def get_reads_covering_positions_data(self, positions, plot=False):
        data = self.data[self.data['reference_index'].isin(positions)]
        plot_data = data.loc[:, ['read_id', 'reference_index', 'variants', 'prob1', 'prob2']]
        pos_n = len(positions)
        select = plot_data.groupby(['read_id']).nunique()
        select.columns = ['id_number', 'reference_index', 'variants', 'prob1', 'prob2']
        a = select[select['reference_index'] == pos_n]
        target_ids = list(a.index.values)
        d = {}
        for pos in positions:
            d[str(pos)] = []
        for i in target_ids:
            r = (plot_data.loc[plot_data['read_id'] == i]).set_index('reference_index')
            for index, row in r.iterrows():
                d[str(index)].append(r.at[index, 'prob2'])
        df_plot = pd.DataFrame(list(zip(target_ids)))
        df_plot.columns = ['read_id']
        for key in d:
            col_val = ''
            col_val += 'P' + ' ' + str(key)
            df_plot[col_val] = d[key]
        if plot:
            del df_plot['read_id']
        return df_plot

    ##Spectral clustering

    def spectral_clusters(self, positions):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        dists = squareform(pdist(X.values))  # ndarray
        neighbor_n = int(math.sqrt(len(positions)))
        knn_distances = np.sort(dists, axis=0)[neighbor_n]  # indexer is usually sqrt(number_of_data_points)
        knn_distances = knn_distances[np.newaxis].T
        local_scale = knn_distances.dot(knn_distances.T)
        a_matrix = dists * dists
        a_matrix = -a_matrix / local_scale
        a_matrix[np.where(np.isnan(a_matrix))] = 0.0
        a_matrix = np.exp(a_matrix)
        np.fill_diagonal(a_matrix, 0)  # now have a_matrix
        A = a_matrix
        L = csgraph.laplacian(A, normed=True)
        n_components = A.shape[0]
        eigenvalues, eigenvectors = eigsh(L, k=n_components, which='LM', sigma=1.0, maxiter=500)
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.xlabel('K clusters')
        plt.ylabel('Eigengap')
        axes = plt.gca()
        axes.set_xlim([-5, 80])
        return plt.grid()

    def spectral_affinity(self, positions, n_clusters):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        spectral_model_rbf = SpectralClustering(n_clusters=n_clusters, affinity='rbf')  # with any value for n_clusters
        labels_rbf = spectral_model_rbf.fit_predict(X)
        spectral_model_nn = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        labels_nn = spectral_model_nn.fit_predict(X)
        affinity = ['rbf', 'nearest_neighbors']
        s_scores = [silhouette_score(X, labels_rbf), silhouette_score(X, labels_nn)]
        plt.bar(affinity, s_scores)
        plt.xlabel('Affinity')
        plt.ylabel('Silhouette Score')
        plt.title('Comparison of different Clustering Models')
        return plt.show()

    def spectral_clustering(self, positions, n_clusters=1, affinity=1, max_number_clusters=None, cluster_size=None,
                            eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return SpectralClustering(n_clusters=n_clusters, affinity=affinity).fit(X), SpectralClustering(
            n_clusters=n_clusters, affinity=affinity).fit_predict(X)

    ##mean shift

    def mean_shift(self, positions, n_clusters=None, affinity=None, max_number_clusters=None, cluster_size=None,
                   eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        quantile_data = X.to_numpy()
        quantile_val = np.quantile(quantile_data, 0.5)
        bandwidth = estimate_bandwidth(X, quantile=quantile_val)
        ms = MeanShift(bandwidth=bandwidth)
        mf = ms.fit(X)
        print('Number of clusters found: ', len(mf.cluster_centers_))
        return mf, ms.fit_predict(X)

    ##K means

    def k_means_clusters(self, positions, max_number_clusters):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, max_number_clusters + 1))
        v = visualizer.fit(X)
        v.show()
        return KMeans(n_clusters=v.elbow_value_).fit(X)

    def k_means(self, positions, n_clusters=None, affinity=None, max_number_clusters=1, cluster_size=None,
                eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, max_number_clusters + 1))
        v = visualizer.fit(X)
        v.show()
        return KMeans(n_clusters=v.elbow_value_).fit(X), KMeans(n_clusters=v.elbow_value_).fit_predict(X)

    ##HDBSCAN

    def HDBSCAN(self, positions, n_clusters=None, affinity=None, max_number_clusters=None, cluster_size=1,
                eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return hdbscan.HDBSCAN(min_cluster_size=cluster_size, gen_min_span_tree=True).fit(X), hdbscan.HDBSCAN(
            min_cluster_size=cluster_size, gen_min_span_tree=True).fit_predict(X)

    ##DBSCAN

    def DBSCAN_eps(self, positions):
        X = self.get_reads_covering_positions_data(positions)
        del X['read_id']
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        a = plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel('eps value')
        print(a)
        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        fig = plt.figure()
        knee.plot_knee()
        plt.xlabel('Points')
        plt.ylabel('eps')
        print('Optimal eps value: ', distances[knee.knee])

    def DBSCAN(self, positions, n_clusters=None, affinity=None, max_number_clusters=None, cluster_size=None,
               eps=1, min_samples=1, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit(X).labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: ', n_clusters_)
        return db.fit(X), db.fit_predict(X)

    ##Agglomerative clustering

    def get_dendrogram(self, positions, y1=0, y2=0, y3=0):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        hierarchy.dendrogram(linkage(X, method='single'), labels=X.index)
        left, right = plt.xlim()
        plt.hlines(y1, left, right, linestyles='dashed', label='y1')
        plt.hlines(y2, left, right, linestyles='dashed', label='y2')
        plt.hlines(y3, left, right, linestyles='dashed', label='y3')
        plt.ylabel('Euclidean Distance')
        plt.title('Dendrogram of subunit data')
        plt.text(right, y1, 'y1')
        plt.text(right, y2, 'y2')
        plt.text(right, y3, 'y3')
        return plt.show()

    def agglomerative_clustering(self, positions, n_clusters=1, affinity=None, max_number_clusters=None,
                                 cluster_size=None,
                                 eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return AgglomerativeClustering(n_clusters=n_clusters).fit(X), AgglomerativeClustering(
            n_clusters=n_clusters).fit_predict(X)

    ##affinity propagation

    def affinity_propagation(self, positions, n_clusters=None, affinity=None, max_number_clusters=None,
                             cluster_size=None, eps=None, min_samples=None, n_components=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return AffinityPropagation().fit(X), AffinityPropagation().fit_predict(X)

    ##Gaussian mixture models

    def gaussian_mixture_models_cluster(self, positions, n_clusters=None, affinity=None, max_number_clusters=None,
                                        cluster_size=None, eps=None, min_samples=None, n_components=1):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return GaussianMixture(n_components=n_clusters).fit(X), GaussianMixture(n_components=n_clusters).fit_predict(X)

    ##plot

    def plot_tSNE_reads_covering_positions_data(self, positions, clustering_algorithm, n_clusters=2,
                                                affinity='nearest_neighbors', max_number_clusters=10, cluster_size=10,
                                                eps=2, min_samples=10, n_components=4, figure_path=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(X)
        method_to_call = getattr(self, clustering_algorithm)
        fit, predictor = method_to_call(positions, n_clusters, affinity, max_number_clusters, cluster_size, eps,
                                        min_samples, n_components)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predictor, s=30, cmap='rainbow')
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(str(len(positions)) + ' ' + 'positions' + ' ' + clustering_algorithm)
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path)
        else:
            plt.show()
        return figure_path

    def plot_PCA_reads_covering_positions_data(self, positions, clusters_n, clustering_algorithm):
        pass

    def plot_number_reads_covering_positions(self, contig, figure_path=None, verbose=False):
        data = self.data.loc[self.data['contig'] == contig].set_index('reference_index')
        data_2 = data.loc[:, 'read_id']
        reads_dict = defaultdict(list)
        pos = []
        for index, value in data_2.items():
            reads_dict[str(index)].append(value)
            pos.append(index)
        positions = list(dict.fromkeys(pos))
        positions.reverse()
        n = 0
        d = {}
        for i in positions:
            n += 1
            if n == 1:
                comp = reads_dict[str(i)]
                d[str(i)] = len(comp)
            else:
                unique_val = set(reads_dict[str(i)]).intersection(comp)
                comp = list(unique_val)
                d[str(i)] = len(comp)
        count_reads = pd.Series(d)
        count_reads.plot(title='Reads for ' + contig, legend=False)
        number = count_reads.min()
        if verbose:
            print('Reads covering all positions in ' + str(contig) + ': ' + str(number))
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path)
        else:
            plt.show()
        return figure_path, number
