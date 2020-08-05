import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil

import artm


class BigARTMModel:
 
    def __init__(self, data_path="train.txt", target_folder="batches", batch_size=1000): 
        """ Takes in Vowpal Wabbit formatted file in data_path and creates self.batch_vectorizer
            that forms batches of data in target_folder of sizes batch_size.

            Parameters:
            data_path (str) - location of the input file
            target_folder (str) - name of a folder for saving batches
            batch_size (int) - batch_size
        """
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
                
        self.batch_vectorizer = artm.BatchVectorizer(data_path=data_path, data_format="vowpal_wabbit", 
                                                target_folder=target_folder, batch_size=batch_size)

    def reweight(self, doc_dict):
        """ Reweights modalities so that they will be treated equally by BigArtm model

            Parameters:
            doc_dict (defaultdict) - a dictionary with the structure:
                                        {
                                        document_id_1: modality_name_1: [token1, token2, ... ],
                                                       modality_name_2: [token1, token2, ... ]
                                                       ...
                                        }
        """
        lengths = tuple(
            sum(count for modality, tokens in modalities.items() for token, count in tokens.items()) 
            for doc, modalities in doc_dict.items()
        )
        p_d = np.array(lengths) / sum(lengths)
        weights = np.zeros(self.n_modalities)

        for doc, modalities in doc_dict.items():
            tmp = np.ones(self.n_modalities)
            for idx, (modality, tokens) in enumerate(modalities.items()):
                tmp[idx] = sum(count for token, count in tokens.items())
            ## if a modality does not have any tokens
            tmp = np.array([val if val != 0 else 1e-06 for val in tmp])
            weights += p_d[doc] * tmp ** -1
            
        self.w_modals = dict(zip(self.modalities, weights * 10))

    def train(self, schedule, doc_dict, num_topics=100, n_background=5, seed=12, manual_weights=None, verbose=True):
        """ Trains BigARTM model

            Parameters:
            schedule (list of namedtuples) - step-by-step schedule for running BigArtm
            
            
            Example:
            Action = namedtuple('Action', ['n_epochs', 'setters'])
            ### setters: [function, function name, regularization tau, class_id, bool (if background topic)])

            schedule = [
                Action(
                    n_epochs=5,
                    setters=[(artm.DecorrelatorPhiRegularizer, 'decorrelator_main_phi', 1, None, False)]
                ),
                ....
            ]
            
            doc_dict (defaultdict) - a dictionary, the same as in self.reweight
            num_topics (int) - number of topics to output
            n_background (int) - number of background topics
            manual_weights (dict) - mannually added weights of modalities
            
        """
        modalities = list(doc_dict[0].keys())
        self.modalities, self.n_modalities = modalities, len(modalities)
       
        if manual_weights is not None:
            self.w_modals = manual_weights
        else:
            self.reweight(doc_dict)
                            
        topic_names = ['topic_' + str(i) for i in range(num_topics)]
        model = artm.ARTM(num_topics=num_topics, dictionary=self.batch_vectorizer.dictionary, 
                          cache_theta=True, class_ids=self.w_modals, topic_names=topic_names, seed=seed)
        model.initialize(self.batch_vectorizer.dictionary)

        for action in schedule:
            if action.setters is not None:
                for regularizer, regularizer_name, tau, class_id, backgr in action.setters:
                    try:
                        if backgr:
                            topic_slice = topic_names[:n_background]
                        else:
                            topic_slice = topic_names[n_background:]

                        if class_id is not None:
                            model.regularizers.add(regularizer(name=regularizer_name, tau=tau, 
                                                           topic_names=topic_slice, class_ids=class_id))
                        else:
                            model.regularizers.add(regularizer(name=regularizer_name, tau=tau, 
                                                           topic_names=topic_slice))
                    except AttributeError:
                        model.regularizers[regularizer_name].tau = tau
            model.fit_offline(batch_vectorizer=self.batch_vectorizer, num_collection_passes=action.n_epochs)
            if verbose:
                print(f'trained for {action.n_epochs} iterations')
        
        self.model = model
        
    def transform(self, model, pred_modality=None, return_perplexity=False):
        """ Transforms new data with respect to the trained model - self.model

            Parameters:
            pred_modality (str) - if name of a modality, then the function returns
                                    p(w/d) - distribution of tokens for each document;
                                    if None - makes theta transformation (p(t/d)) of new data
            return_perplexity (bool) - if True, returns perplexity score for pred_modality
        """
        if pred_modality is None:
            return model.transform(batch_vectorizer=self.batch_vectorizer).T
                                
        else:
            if return_perplexity:
                bigartm_train.model.transform(self.batch_vectorizer, predict_class_id=pred_modality, theta_matrix_type=None)
                model.scores.add(artm.PerplexityScore(name='perplexity'))
                return model.get_score('perplexity')
            else:
                return model.transform(self.batch_vectorizer, predict_class_id=pred_modality).T.sort_index()

    def predictCluster(theta, good_topics):
        """ predicts label without considering only list of good topics topics """
        return pd.DataFrame(theta.loc[:, good_topics].idxmax(axis=1), columns=['cluster'])
     
    @staticmethod
    def theta_sparsity(theta, sample_docs=100, top_topics=2, n_background=0, random_state=12):
        """ Parameters:
            theta (pandas.DataFrame) - theta matrix
            sample_docs (int) - number of sampled docs to calculate statistics for
            top_topics (int) - top topics to calculate statistics for
            n_background (int) - number of backround topics that were used for training theta

            Returns:
            mean and standard deviation of top topics for a sample of documents
        """
        sample = theta.sample(n=sample_docs, random_state=random_state).iloc[:, n_background:]
        stats = sample.apply(lambda x: sum(np.sort(x)[-top_topics:]), axis=1)
        if sum(stats.values == 0) > 0:
            return print('Some documents have no topic distribution, please, change model parameters')
        return stats.mean(), stats.std()

    @staticmethod   
    def plot_topic_distr(theta):
        """ Parameters:
            theta (pandas.DataFrame) - theta matrix

            Returns:
            Bar plot of average probabilities for each topic
        """
        stats = theta.mean(0)
        df_tmp = pd.DataFrame(dict(zip(range(len(stats)), stats)), index=[0])

        sns.set()
        sns.barplot(data=df_tmp)

        return plt.show()

    @staticmethod
    def print_phi(phi, topic_name='topic_0', num_tokens=10):
        """ Prints num_tokens token distribution for SELECTED topic for EACH modality

            Parameters:
            phi (pandas.DataFrame) - phi matrix
            num_tokens - number of top tokens to display

            Returns:
            DataFrame with distributions
        """
        modalities = set([ind[0] for ind in phi.index])

        inds = pd.MultiIndex.from_tuples(list(phi.index), names=['mcc', 'token'])
        phi = pd.DataFrame(phi, index=inds)

        tot_list = []
        for mod in modalities:
            df_mod = phi.loc[(mod, slice(None))]
            t = df_mod[[topic_name]].sort_values(topic_name, ascending=False).iloc[:num_tokens]
            df = pd.DataFrame({'top_tokens': list(t.index), 'probab': list(t[topic_name])})
            df.columns = pd.MultiIndex.from_tuples([(mod,'top_tokens'), (mod,'probab')])
            tot_list.append(df)

        return pd.concat(tot_list, axis=1).fillna('---')