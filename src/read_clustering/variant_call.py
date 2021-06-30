#!/usr/bin/env python
"""Variant call file class handler"""
########################################################################
# File: variant_call.py
#
#
# Author: Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar
# History: 06/23/20 Created
########################################################################

import math
import os
import re
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, \
    estimate_bandwidth, KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
from kneed import KneeLocator
from pathlib import Path


# sns.set()


class VariantCall(object):
    """Read in variant call file and give access points to various types of data"""

    def __init__(self, file_path):
        """
        Initialize the class
        :param file_path: path to csv file
        """
        self.data = pd.read_csv(file_path)
        self.canonical = {"A", "C", "G", "T"}

    @staticmethod
    def load_variant_data(variant_csv_path, label):
        assert os.path.exists(variant_csv_path), f"variant_csv_path does not exist: {variant_csv_path}"
        variant_data = pd.read_csv(variant_csv_path)
        variant_data["label"] = label
        return variant_data

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
        :return dataframe with both read_id equal to read_id and reference_index is in positions list
        """
        data = self.data[(self.data['read_id'] == read_id) & (self.data['reference_index'].isin(positions))]
        if data.empty:
            return False
        return data

    def get_contig_positions(self, contig):  # assumes there is at least one read that covers all positions
        data = self.data.loc[self.data['contig'] == contig]
        data_2 = data.groupby(['reference_index']).nunique()
        positions = []
        for i, row in data_2.iterrows():
            positions.append(i)
        return positions

    def get_reads_covering_positions_data(self, positions, plot=False):
        data = self.data[self.data['reference_index'].isin(positions)]
        plot_data = data.loc[:, ['read_id', 'reference_index', 'variants', 'prob1', 'prob2']]
        pos_n = len(positions)
        select = plot_data.groupby(['read_id']).nunique()
        select.rename(columns={'read_id': 'id_number'}, inplace=True)
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
            col_val = str(key)
            df_plot[col_val] = d[key]
        if plot:
            del df_plot['read_id']
        return df_plot

    # Spectral clustering

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
        s_scores = [metrics.silhouette_score(X, labels_rbf), metrics.silhouette_score(X, labels_nn)]
        plt.bar(affinity, s_scores)
        plt.xlabel('Affinity')
        plt.ylabel('Silhouette Score')
        plt.title('Comparison of different Clustering Models')
        return plt.show()

    def spectral_clustering(self, positions, n_clusters=1, affinity='x'):
        # needs n_clusters, affinity as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return SpectralClustering(n_clusters=n_clusters, affinity=affinity).fit(X), SpectralClustering(
            n_clusters=n_clusters, affinity=affinity).fit_predict(X)

    # mean shift

    def mean_shift(self, positions, find_optimal=True):
        # need no **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        if find_optimal:
            quantile_data = X.to_numpy()
            quantile_val = np.quantile(quantile_data, 0.5)
            bandwidth = estimate_bandwidth(X, quantile=quantile_val)
        else:
            bandwidth = 0.5
        print(f"Bandwith selected: {bandwidth}")
        ms = MeanShift(bandwidth=bandwidth)
        mf = ms.fit(X)
        print('Number of clusters found: ', len(mf.cluster_centers_))
        return mf, ms.fit_predict(X)

    # K means

    def k_means_clusters(self, positions, max_number_clusters):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, max_number_clusters + 1))
        v = visualizer.fit(X)
        v.show()
        return v.elbow_value_

    def k_means(self, positions, max_number_clusters=1, find_optimal=True):
        # needs max_number_clusters as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        if find_optimal:
            n = self.k_means_clusters(positions, max_number_clusters)
        else:
            n = max_number_clusters
        return KMeans(n_clusters=n).fit(X), KMeans(n_clusters=n).fit_predict(X)

    # HDBSCAN

    def HDBSCAN(self, positions, cluster_size=1):
        # needs cluster_size as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return hdbscan.HDBSCAN(min_cluster_size=cluster_size, gen_min_span_tree=True).fit(X), hdbscan.HDBSCAN(
            min_cluster_size=cluster_size, gen_min_span_tree=True).fit_predict(X)

    # DBSCAN

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
        plt.figure()
        knee.plot_knee()
        plt.xlabel('Points')
        plt.ylabel('eps')
        print('Optimal eps value: ', distances[knee.knee])

    def DBSCAN(self, positions, eps=1, min_samples=1):
        # needs eps, min_samples as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit(X).labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: ', n_clusters_)
        return db.fit(X), db.fit_predict(X)

    # Agglomerative clustering

    def get_dendrogram(self, positions, y1=0, y2=0, y3=0, figure_path=None):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        hierarchy.dendrogram(hierarchy.linkage(X, method='single'), labels=list(X.index.values))
        left, right = plt.xlim()
        plt.hlines(y1, left, right, linestyles='dashed', label='y1')
        plt.hlines(y2, left, right, linestyles='dashed', label='y2')
        plt.hlines(y3, left, right, linestyles='dashed', label='y3')
        plt.ylabel('Euclidean Distance')
        plt.title('Dendrogram of subunit data')
        plt.text(right, y1, 'y1')
        plt.text(right, y2, 'y2')
        plt.text(right, y3, 'y3')
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path)
        else:
            plt.show()
        return figure_path

    def agglomerative_clustering(self, positions, n_clusters=1):
        # needs n_clusters as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return AgglomerativeClustering(n_clusters=n_clusters).fit(X), AgglomerativeClustering(
            n_clusters=n_clusters).fit_predict(X)

    # affinity propagation

    def affinity_propagation(self, positions):
        # needs no **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return AffinityPropagation().fit(X), AffinityPropagation().fit_predict(X)

    # Gaussian mixture models

    def SelBest(self, arr: list, X: int) -> list:
        dx = np.argsort(arr)[:X]
        return arr[dx]

    def silhouette_score(self, positions):
        X = self.get_reads_covering_positions_data(positions)
        del X['read_id']
        n_clusters = np.arange(2, 20)
        sils = []
        sils_err = []
        iterations = 20
        for n in n_clusters:
            tmp_sil = []
            for _ in range(iterations):
                gmm = GMM(n, n_init=2).fit(X)
                labels = gmm.predict(X)
                sil = metrics.silhouette_score(X, labels, metric='euclidean')
                tmp_sil.append(sil)
            val = np.mean(self.SelBest(list(tmp_sil), int(iterations / 5)))
            err = np.std(tmp_sil)
            sils.append(val)
            sils_err.append(err)
        plt.errorbar(n_clusters, sils, yerr=sils_err)
        plt.title("Silhouette Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        return plt.show()

    def BIC(self, positions):
        X = self.get_reads_covering_positions_data(positions)
        del X['read_id']
        n_clusters = np.arange(2, 20)
        bics = []
        bics_err = []
        iterations = 20
        for n in n_clusters:
            tmp_bic = []
            for _ in range(iterations):
                gmm = GMM(n, n_init=2).fit(X)
                tmp_bic.append(gmm.bic(X))
            val = np.mean(self.SelBest(np.array(tmp_bic), int(iterations / 5)))
            err = np.std(tmp_bic)
            bics.append(val)
            bics_err.append(err)
        plt.errorbar(n_clusters, bics, yerr=bics_err, label='BIC')
        plt.title("BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

        plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
        plt.title("Gradient of BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("grad(BIC)")
        plt.legend()

    def gaussian_mixture_models(self, positions, n_clusters=1):
        # needs n_clusters as **other_params
        X = self.get_reads_covering_positions_data(positions, plot=True)
        return GMM(n_components=n_clusters).fit(X), GMM(n_components=n_clusters).fit_predict(X)

    # plot

    def plot_tSNE_reads_covering_positions_data(self, positions, clustering_algorithm, figure_path=None,
                                                **other_params):
        # **other_params can be : n_clusters=,affinity=, max_number_clusters=, cluster_size=,eps=, min_samples=,
        # n_components=, find_optimal=,  according to what each clustering method requires
        X = self.get_reads_covering_positions_data(positions, plot=True)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(X)
        method_to_call = getattr(self, clustering_algorithm)
        fit, predictor = method_to_call(positions, **other_params)
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

    def plot_UMAP_reads_covering_positions_data(self, positions, clustering_algorithm, figure_path=None, W=None,
                                                **other_params):
        # **other_params can be : n_clusters=,affinity=, max_number_clusters=, cluster_size=,eps=, min_samples=,
        # n_components=, find_optimal=,  according to what each clustering method requires
        X = self.get_reads_covering_positions_data(positions, plot=True)
        if W is not None:
            X = np.multiply(X, W)
        reducer = umap.UMAP()
        umap_results = reducer.fit_transform(X)
        method_to_call = getattr(self, clustering_algorithm)
        fit, predictor = method_to_call(positions, **other_params)
        plt.scatter(umap_results[:, 0], umap_results[:, 1], c=predictor, s=30, cmap='rainbow')
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(str(len(positions)) + ' ' + 'positions' + ' ' + clustering_algorithm)
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path)
        else:
            plt.show()
        return figure_path

    def plot_PCA_reads_covering_positions_data(self, positions, clustering_algorithm, figure_path=None, **other_params):
        X = self.get_reads_covering_positions_data(positions, plot=True)
        scaler = StandardScaler()
        scaler.fit(X)
        scaled_data = scaler.transform(X.values)
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        method_to_call = getattr(self, clustering_algorithm)
        fit, predictor = method_to_call(positions, **other_params)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], s=40)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=predictor, s=40, cmap='rainbow')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(str(len(positions)) + ' ' + 'positions' + ' ' + clustering_algorithm + ' ' + 'clustering')
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path)
        else:
            plt.show()
        return figure_path

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

    def clusters_for_position_pdf(self, positions, pos_to_evaluate, max_number_clusters, find_optimal=False,
                                  cluster_count=False):
        """Plot the PDF for the reads in the clusters of each positions given 
        :param positions: complete list of the reference_index of the subunit 
        :param pos_to_evaluate: list of reference_index to obtain their density plots
        :param max_number_clusters: maximum number of clusters to be considered by the "elbow method", or 
         clusters to be found by kmeans if find_optimal = False.
        :param find_optimal: if True runs "elbow method" automatically to find optimal number of clusters to be
        found by kmeans
        :param cluster_count: verbose option
        :return Density plot of the number of reads (y axis) and variant probability (x axis) for the clusters in 
        each pos_to-evaluate
        """
        fit, predict = self.k_means(positions, max_number_clusters=max_number_clusters, find_optimal=find_optimal)
        indexes = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
        X = self.get_reads_covering_positions_data(positions, plot=True)
        possible_colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'magenta', 'cyan', 'brown', 'gray']
        for pos in pos_to_evaluate:
            for key, value in indexes.items():
                if cluster_count:
                    print(str(key) + ' data point count: ', len(value))
                X_new = X.iloc[value.tolist()]
                X_pos = X_new.loc[:, str(pos)].T
                grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20)
                grid.fit(X_pos[:, None])
                bw = grid.best_params_
                kde_skl = KernelDensity(bandwidth=bw['bandwidth'])
                kde_skl.fit(X_pos[:, None])
                x_grid = np.linspace(-.3, 1.3, 1000)
                log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
                plt.plot(x_grid, np.exp(log_pdf), color=possible_colors[key], alpha=0.5, lw=1.5, label=str(key))

            plt.legend(title='Cluster (kmeans)')
            plt.title('Position ' + str(pos) + ' density plot')
            plt.xlabel('Variant probability')
            plt.ylabel('Density (reads)')
            plt.show()
        return

    def kmeans_clusters_for_position_KS_test(self, positions, pos_to_evaluate, max_number_clusters, find_optimal=False):
        """Pairwise KS test comparing the clusters found by kmeans on a specific position to determine if they 
        are significantly different  
        :param positions: complete list of the reference_index of the subunit to be evaluated
        :param pos_to_evaluate: reference_index whose distribution across the clusters will be compared 
        :param max_number_clusters: maximum number of clusters to be considered by the "elbow method", or 
         clusters to be found by kmeans if find_optimal = False.
        :param find_optimal: if True runs "elbow method" automatically to find optimal number of clusters to be
        found by kmeans
        :return ks_df: dataframe with the p value resulting from the pairwise KS test between cluster pairs
        :return min_p_val: minimum p value found in ks_df
        """
        fit, predict = self.k_means(positions, max_number_clusters=max_number_clusters, find_optimal=find_optimal)
        indexes = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
        X = self.get_reads_covering_positions_data(positions, plot=True)
        ks_compare = []
        d = {}
        critical_p = 1.36 / np.sqrt(len(X.index))
        for key, value in indexes.items():
            X_new = X.iloc[value.tolist()]
            X_pos = X_new.loc[:, pos_to_evaluate].T
            grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20)
            grid.fit(X_pos[:, None])
            bw = grid.best_params_
            kde_skl = KernelDensity(bandwidth=bw['bandwidth'])
            kde_skl.fit(X_pos[:, None])
            x_grid = np.linspace(-.3, 1.3, 1000)
            ks_compare.append(kde_skl.score_samples(x_grid[:, np.newaxis]))
            d[str(kde_skl.score_samples(x_grid[:, np.newaxis]))] = key

        ks_df = pd.DataFrame([], columns=list(range(0, len(d))), index=list(range(0, len(d))))
        min_p_val = (ks_df.min()).min()
        return ks_df, min_p_val

    def positions_modification_plot_kmeans_clusters(self, positions, max_number_clusters, find_optimal=False,
                                                    cluster_count=False, subunit_name='', threshold=0.5,
                                                    pseudou="ql", twoprimeo=["na", "ob", "pc", "qd"]):
        fit, predict = self.k_means(positions, max_number_clusters=max_number_clusters, find_optimal=find_optimal)
        indexes = {i: np.where(fit.labels_ == i)[0] for i in range(fit.n_clusters)}
        X = self.get_reads_covering_positions_data(positions, plot=True)
        color_df = pd.DataFrame([], index=list(range(0, len(indexes))), columns=positions)
        # str_pos = [str(x) for x in positions]
        # mod_df = pd.DataFrame([], index=list(range(0, len(indexes))), columns=str_pos)
        for key, value in indexes.items():
            if cluster_count:
                print('Data points in cluster' + str(key) + ' : ', len(value))
            X_cluster = X.iloc[value.tolist()]
            for pos in positions:
                X_pos = X_cluster.loc[:, str(pos)]
                X_modified = X_pos[X_pos > threshold]
                color_df.at[[key], [pos]] = (len(X_modified) / len(X_pos)) * 100

        plt.figure(figsize=(18, 5))
        ax = sns.heatmap(color_df.astype(float), xticklabels=True, annot=False, vmin=0, vmax=100, cmap="OrRd")
        ax.hlines(list(range(0, len(indexes))), *ax.get_xlim(), linewidth=5, color='white')
        ax.set_title('Modification occurrence in ' + subunit_name + ' positions', fontsize=18)
        ax.set_xlabel("Positions (5' to 3')", fontsize=15)
        ax.set_ylabel('Cluster (kmeans)', fontsize=15)
        pseduo_u_df = self.get_positions_of_variant_set(pseudou)
        twoprimeo_df = self.get_positions_of_variant_sets(twoprimeo)
        pseduo_u_pos = pseduo_u_df[pseduo_u_df["contig"] == subunit_name]["reference_index"].values
        twoprimeo_pos = twoprimeo_df[twoprimeo_df["contig"] == subunit_name]["reference_index"].values
        [t.set_color('red') for t in ax.xaxis.get_ticklabels() if int(t.get_text()) in pseduo_u_pos]
        [t.set_color('blue') for t in ax.xaxis.get_ticklabels() if int(t.get_text()) in twoprimeo_pos]
        return plt.show()


class VariantCalls(VariantCall):
    """Read in variant call file and give access points to various types of data"""

    def __init__(self, file_paths, labels, color_map="tab20"):
        super().__init__(file_paths[0])
        data = []
        self.labels = labels
        self.file_paths = file_paths
        for path, label in zip(file_paths, labels):
            data.append(self.load_variant_data(path, label))
        self.data = pd.concat(data, ignore_index=True)
        self.experiments = sorted(labels)
        self.color_map = dict(zip(self.experiments, sns.color_palette(color_map)))
        self.pseduo_u_pos = [775, 959, 965, 985, 989, 1003, 1041, 1051, 1055, 1109, 1123,
                             2128, 2132, 2190, 2257, 2259, 2263, 2265, 2313, 2339, 2348, 2350,
                             2415, 2734, 2825, 2864, 2879, 2922, 2943, 2974, 105, 119, 210, 301, 465, 631, 758, 765,
                             998, 1180, 1186,
                             1289, 1414]
        self.twoprimeo_pos = [648, 649, 662, 804, 806, 816, 866, 875, 897, 907, 1132,
                              1436, 1448, 1449, 1887, 2196, 2219, 2255, 2279, 2280, 2287, 2336,
                              2346, 2416, 2420, 2618, 2639, 2723, 2728, 2790, 2792, 2814, 2920,
                              2921, 2945, 2947, 2958, 27, 99, 413, 419, 435, 540, 561, 577, 618, 795, 973,
                              1006, 1125, 1268, 1270, 1427, 1571, 1638]

    def plot_UMAP_by_label(self, contig, positions, figure_path=None, n_components=2, n=None, legend=True, **other_params):
        data = self.data[(self.data["contig"] == contig) & (self.data['reference_index'].isin(positions))]
        df = data.pivot(index=['label', 'read_id'], columns=['reference_index'], values='prob2')
        X = df.dropna()
        if n is not None:
            df = []
            for x in self.experiments:
                a = X.loc[x][:n]
                a["label"] = x
                a["read_id"] = a.index
                df.append(a.set_index(["label", "read_id"]))
            X = pd.concat(df)

        reducer = umap.UMAP(n_components=n_components)
        umap_results = reducer.fit_transform(X)
        X["umap_result_x"] = umap_results[:, 0]
        X["umap_result_y"] = umap_results[:, 1]
        #         predictor = hdbscan.HDBSCAN(min_cluster_size=cluster_size, gen_min_span_tree=True).fit_predict(X)
        #         X["predictor"] = predictor
        markers = mpl.markers.MarkerStyle.markers.keys()
        marker = 'o'
        #         colors = ('g', 'r', 'c', 'm', 'y', 'k', 'w', 'b')

        fig = plt.figure(figsize=(15, 15))
        if n_components == 2:
            ax = fig.add_subplot(111)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            X["umap_result_z"] = umap_results[:, 2]
        ax.set_facecolor("white")

        # for item in [fig, ax]:
        #     item.patch.set_visible(False)

        for experiment in self.experiments:
            plot_data = X.xs(experiment, level="label")
            #     plt.scatter(plot_data["umap_result_x"].values, plot_data["umap_result_y"].values, marker=marker, s=30, c=plot_data["predictor"].values, cmap='rainbow')
            if n_components == 3:
                ax.scatter(plot_data["umap_result_x"].values, plot_data["umap_result_y"].values,
                           plot_data["umap_result_z"].values, color=self.color_map[experiment], marker=marker,
                           label=experiment, **other_params)
            else:
                ax.scatter(plot_data["umap_result_x"].values, plot_data["umap_result_y"].values,
                           color=self.color_map[experiment], marker=marker, label=experiment, **other_params)

                #             plt.scatter(plot_data["umap_result_x"].values, plot_data["umap_result_y"].values, marker=marker, s=3, c=color, cmap='rainbow', label=experiment)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        if n_components == 3:
            ax.set_zlabel("UMAP 3")
        if legend:
            plt.legend()

        plt.title(f"{contig}: {str(len(positions))} positions")
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path, dpi=1000)
        else:
            plt.show()

    def plot_all_heatmap_dendrograms(self, output_dir, labels=None, n=None, col_cluster=False,
                                     method='average', metric='correlation', row_cluster=True,
                                     pseduo_u_pos=None, twoprimeo_pos=None, legend=True):
        """Plot all clustering heatmaps for each experiment and save to directory
        :param legend: boolean option to plot legend
        :param n: number of reads to include in clustering
        :param col_cluster: bool, cluster columns
        :param method: clustering method
        :param metric: clustering metric
        :param row_cluster: boolean option to cluster rows
        :param twoprimeo_pos: optional list of twoprimeo positions
        :param pseduo_u_pos: optional list of pseudoU positions
        :param output_dir: output directory
        :param labels: list of labels to cluster and plot
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if labels is None:
            labels = self.labels
        contig_pos = [[contig, self.get_contig_positions(contig)] for contig in set(self.data["contig"])]
        for label in labels:
            for contig, positions in contig_pos:
                X = self.get_X(contig, positions, label=label)
                figure_path = os.path.join(output_dir, f"{label}_{contig}_{n}_{metric}.png")
                self._plot_heatmap_dendrogram(X, n=n, col_cluster=col_cluster, figure_path=figure_path,
                                              method=method, metric=metric, row_cluster=row_cluster,
                                              pseduo_u_pos=pseduo_u_pos, twoprimeo_pos=twoprimeo_pos, legend=legend)

    def _plot_heatmap_dendrogram(self, X, n=100, col_cluster=True, figure_path=None,
                                 method='average', metric='euclidean', row_cluster=True,
                                 pseduo_u_pos=None, twoprimeo_pos=None, legend=True):
        if n is not None:
            df = []
            for x in self.experiments:
                a = X.xs(x, level="label")[:n]
                a["label"] = x
                a["read_id"] = a.index
                df.append(a.set_index(["read_id", "label"]))
            X = pd.concat(df)

        row_colors = pd.DataFrame(X.index.get_level_values(1))["label"].map(self.color_map)
        data = X.reset_index(drop=True)

        g = sns.clustermap(data, method=method, metric=metric, row_colors=row_colors, col_cluster=col_cluster,
                           row_cluster=row_cluster,
                           yticklabels=False, xticklabels=True, cmap="OrRd", figsize=(20, 20))

        if not legend:
            g.cax.set_visible(False)
        ax = g.ax_heatmap

        if pseduo_u_pos is None:
            pseduo_u_pos = self.pseduo_u_pos
        if twoprimeo_pos is None:
            twoprimeo_pos = self.twoprimeo_pos

        [t.set_color('red') for t in ax.xaxis.get_ticklabels() if int(t.get_text()) in pseduo_u_pos]
        [t.set_color('blue') for t in ax.xaxis.get_ticklabels() if int(t.get_text()) in twoprimeo_pos]

        experiment_labels = []
        for experiment, color in self.color_map.items():
            red_patch = mpatches.Patch(color=color, label=experiment)
            experiment_labels.append(red_patch)

        red_pseudoU = mpatches.Patch(color="red", label="Pseudouridine")
        blue_twoprime = mpatches.Patch(color="blue", label="2'O methylcytosine")

        if legend:
            first_legend = plt.legend(handles=experiment_labels, bbox_to_anchor=(1.5, 1.2), loc='upper left', ncol=1,
                                      title="Experiments")
            plt.gca().add_artist(first_legend)

            plt.legend(handles=[red_pseudoU, blue_twoprime], bbox_to_anchor=(1.5, .5), loc='upper left',
                       title="Modifications")
        # h = [plt.plot([],[], color="gray", marker="o", ms=i, ls="")[0] for i in range(5,13)]
        # plt.legend(handles=h, labels=range(5,13),loc=(1.03,0.5), title="Quality")
        labels = [str(int(item.get_text()) + 1) for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path, dpi=1000)
        else:
            plt.show()

    def plot_heatmap_dendrogram(self, contig, positions, label=None, n=100, col_cluster=True,
                                figure_path=None, method='average', metric='euclidean', row_cluster=True,
                                pseduo_u_pos=None, twoprimeo_pos=None, legend=True):
        X = self.get_X(contig, positions, label=label)
        if pseduo_u_pos is None:
            pseduo_u_pos = self.pseduo_u_pos
        if twoprimeo_pos is None:
            twoprimeo_pos = self.twoprimeo_pos
        # g = sns.heatmap(X[:n], xticklabels=True, yticklabels=False, annot=False, cmap="OrRd")
        self._plot_heatmap_dendrogram(X, n=n, col_cluster=col_cluster, figure_path=figure_path,
                                      method=method, metric=metric, row_cluster=row_cluster,
                                      pseduo_u_pos=pseduo_u_pos, twoprimeo_pos=twoprimeo_pos, legend=legend)

    def get_X(self, contig, positions, label=None):
        data = self.data[(self.data["contig"] == contig) & (self.data['reference_index'].isin(positions))]
        df = data.pivot(index=['read_id', 'label'], columns=['reference_index'], values='prob2')
        X = df.dropna()
        X = X.sort_index()
        if label is not None:
            assert label in self.labels, f"Input label [{label}] is not in {self.labels}"
            # X = X.xs(label, level="label")
            X = X.loc[(slice(None), label), :]
        return X

    def plot_ld_heatmap(self, contig, positions, stat="r2", cmap="OrRd", norm=None, linewidths=0,
                        figure_path=None, pseduo_u_pos=None, twoprimeo_pos=None, vmax=None):
        options = ["r2", "D", "D'"]
        assert stat in options, f"Stat {stat} cannot be found. Must select from: {options}"
        data = self.data[(self.data["contig"] == contig) & (self.data['reference_index'].isin(positions))]
        df = data.pivot(index=['read_id', 'label'], columns=['reference_index'], values='prob2')
        X = df.dropna()
        return self._plot_ld_heatmap(X, stat, cmap, norm, linewidths, figure_path,
                                     pseduo_u_pos=pseduo_u_pos, twoprimeo_pos=twoprimeo_pos, vmax=vmax)

    def _plot_ld_heatmap(self, X, stat="r2", cmap="OrRd", norm=None, linewidths=0,
                         figure_path=None, pseduo_u_pos=None, twoprimeo_pos=None, vmax=None):
        d_stats = self.get_correlation(X, stat)
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(d_stats, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(16, 14))
        # Generate a custom diverging colormap
        #         im = ax.imshow(data, cmap=plt.cm.hot, interpolation='none', vmax=threshold)
        #         cbar = fig.colorbar(im, extend='max')
        #         cbar.cmap.set_over('green')
        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(d_stats, mask=mask, cmap=cmap, yticklabels=True, xticklabels=True,
                         square=True, linewidths=linewidths, cbar_kws={"shrink": .5}, norm=norm, vmax=vmax)  # vmax
        if pseduo_u_pos is None:
            pseduo_u_pos = self.pseduo_u_pos
        if twoprimeo_pos is None:
            twoprimeo_pos = self.twoprimeo_pos
        [t.set_color('red') for t in ax.xaxis.get_ticklabels() if
         int(re.search(r'\d+', t.get_text()).group()) in pseduo_u_pos]
        [t.set_color('blue') for t in ax.xaxis.get_ticklabels() if
         int(re.search(r'\d+', t.get_text()).group()) in twoprimeo_pos]
        [t.set_color('red') for t in ax.yaxis.get_ticklabels() if
         int(re.search(r'\d+', t.get_text()).group()) in pseduo_u_pos]
        [t.set_color('blue') for t in ax.yaxis.get_ticklabels() if
         int(re.search(r'\d+', t.get_text()).group()) in twoprimeo_pos]
        ax.set_xticks(ax.get_yticks())
        ax.set_xticklabels([int(x.get_text()) + 1 for x in ax.get_yticklabels()])
        ax.set_yticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_xticklabels())

        #         experiment_labels = []
        #         for experiment, color in self.color_map.items():
        #             red_patch = mpatches.Patch(color=color, label=experiment)
        #             experiment_labels.append(red_patch)

        red_pseudoU = mpatches.Patch(color="red", label="Pseudouridine")
        blue_twoprime = mpatches.Patch(color="blue", label="2'O methylcytosine")
        plt.legend(handles=[red_pseudoU, blue_twoprime], bbox_to_anchor=(0, 1), loc='lower right',
                   title="Modifications")
        if figure_path is not None:
            assert not os.path.exists(figure_path), "Save fig path does exist: {}".format(figure_path)
            plt.savefig(figure_path, dpi=1000)
        else:
            plt.show()

        return d_stats

    def get_r2(self, a, b):
        return self.get_corr(a, b, stat="r2")

    def get_d(self, a, b):
        return self.get_corr(a, b, stat="D")

    def get_d_prime(self, a, b):
        return self.get_corr(a, b, stat="D'")

    @staticmethod
    def get_corr(a, b, stat="r2"):
        joint_AB = np.mean((a >= 0.5) & (b >= 0.5))
        pA = (a >= 0.5).mean()
        pB = (b >= 0.5).mean()
        pa = (a < 0.5).mean()
        pb = (b < 0.5).mean()
        D = joint_AB - (pA * pB)
        if D > 0:
            Dmax = min([pa * pB, pA * pb])
        if D < 0:
            Dmax = min([pA * pB, pa * pb]) * -1
        if D == 0:
            D_prime = 0
        else:
            D_prime = D / Dmax
        denom = (pA * pB * pa * pb)
        if denom == 0:
            coef_of_corr = 0
        else:
            coef_of_corr = (D ** 2) / (pA * pB * pa * pb)

        if stat == "D":
            return D
        elif stat == "D'":
            return D_prime
        elif stat == "r2":
            return coef_of_corr

    def write_correlations(self, output_path, stat="r2", labels=None):
        """Write out correlations within each experiment to a csv file

        :param output_path: path to output file
        :param stat: optional statistic to use
        :param labels: optional labels to use instead of all experiments
        """
        data = self.get_experiment_correlations(stat=stat, labels=labels)
        data.to_csv(output_path, index=False)
        return True

    def get_method(self, stat):
        if stat == "r2":
            return self.get_r2
        if stat == "D":
            return self.get_d
        if stat == "D'":
            return self.get_d_prime
        else:
            return stat

    def get_correlation(self, X, stat="r2"):
        return X.corr(self.get_method(stat))

    def get_experiment_correlations(self, stat="r2", labels=None):
        """Get correlations of each experiment

        :param stat: optional statistic to use
        :param labels: optional labels to use instead of all experiments
        """
        if labels is None:
            labels = self.labels

        data = None
        order = ['ref_index1', 'ref_index2', "contig"]
        contig_pos = [[contig, self.get_contig_positions(contig)] for contig in set(self.data["contig"])]
        for label in labels:
            order.append(label)
            label_df = None
            for contig, positions in contig_pos:
                X = self.get_X(contig, positions, label=label)
                correlations = self.get_correlation(X, stat)
                df = correlations.rename_axis(None).rename_axis(None, axis=1)
                df_corr = df.stack().reset_index()
                df_corr.columns = ['ref_index1', 'ref_index2', label]
                mask_dups = (df_corr[['ref_index1', 'ref_index2']].apply(frozenset, axis=1).duplicated()) | (
                        df_corr['ref_index1'] == df_corr['ref_index2'])
                df_corr = df_corr[~mask_dups]
                df_corr["contig"] = contig
                if label_df is None:
                    label_df = df_corr
                else:
                    label_df = pd.concat([label_df, df_corr])

            if data is None:
                data = label_df
            else:
                data = pd.merge(data, label_df, how="outer", on=['ref_index1', 'ref_index2', "contig"])
        return data[order]

    def get_experiment_percent_modified(self, labels=None):
        """Get percent modified of each position for each experiment

        :param labels: optional labels to use instead of all experiments
        """
        order = ["contig"]
        if labels is None:
            labels = self.labels
        data = None
        contig_pos = [[contig, self.get_contig_positions(contig)] for contig in set(self.data["contig"])]
        for label in labels:
            order.append(label)
            label_df = None
            for contig, positions in contig_pos:
                X = self.get_X(contig, positions, label=label)
                probs = pd.DataFrame((X >= 0.5).mean()).rename(columns={0: label})
                probs["contig"] = contig
                if label_df is None:
                    label_df = probs
                else:
                    label_df = pd.concat([label_df, probs])
            if data is None:
                data = label_df
            else:
                data = pd.merge(data, label_df, how="outer", on=["reference_index", "contig"])
        data = data[order]
        return data

    def write_experiment_percent_modified(self, output_path, labels=None):
        """Write percent modified of each position for each experiment

        :param output_path: path to output file
        :param labels: optional labels to use instead of all experiments
        """
        self.get_experiment_percent_modified(labels=labels).to_csv(output_path)
        return True
