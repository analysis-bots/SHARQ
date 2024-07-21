import itertools

from .Utilities import *
from .RulesUtils import *


class Sharq:
    def __init__(self, dataset=None, max_bins_size=4, score_func=is_score_function, i_func=top_1_i_function, min_support=0.05, lift_limit=0.05, sample_size=10000, operator=Operators.OPT_MIN, rules_set=None, size=15):
        self.bins_df = None
        self.bins_rules_dict = None
        self.dataset = dataset
        self.max_bins_size = max_bins_size
        self.score_func = score_func
        self.i_func = i_func
        self.min_support = min_support
        self.lift_limit = lift_limit
        self.operator = operator
        self.shap_scores = None
        self.size = size
        self.dataset = dataset
        self.avg_bins_df = None
        if rules_set is None and dataset is not None:
            self.binary_df, val_to_bins_dict, self.avg_bins_df, self.elements_support_dict = dataset_prep_by_hyper_parameters(
                dataset, max_bins_size, sample_size)
            self.rules_set = rules_prep_by_hyper_parameters(self.binary_df, score_func, min_support, lift_limit)
        else:
            self.rules_set = rules_set
            if score_func is not is_score_function:
                self.rules_set['score'] = self.rules_set.apply(lambda row: score_func(row), axis=1)
        self.elements = get_rules_set_elements(self.rules_set)
        self.rules_set = add_elements_columns(self.rules_set)

    def run_sharq(self, rules_set=None):
        if rules_set is not None:
            rules = rules_set
        else:
            rules = self.rules_set
        self.shap_scores = \
            calc_short_shap_by_all_elements(rules, size=self.size, operator=self.operator, i_func=self.i_func, elements=self.elements)
        # convert the shap scores to dict and sort it by the shap score
        self.shap_scores = dict((key, value) for key, value in self.shap_scores)
        self.shap_scores = dict(sorted(self.shap_scores.items(), key=lambda item: item[1], reverse=False))
        return self.shap_scores

    def set_orig_dataset(self, dataset, sample_size=10000, max_bins_size=4):
        self.dataset = dataset
        self.binary_df, val_to_bins_dict, self.avg_bins_df, self.elements_support_dict = dataset_prep_by_hyper_parameters(
            dataset, max_bins_size, sample_size)
        self.bins_df = pd.DataFrame()
        for column in self.avg_bins_df.columns:
            if self.avg_bins_df[column].dtype == float:
                self.bins_df[column] = self.avg_bins_df[column].apply(lambda x: val_to_bins_dict[column + '_' + str(x).split('.')[0]])
            else:
                self.bins_df[column] = self.avg_bins_df[column]

    def get_element_frequency(self, element):
        return self.elements_support_dict[element] * 100

    def get_elements_frequency(self):
        elements_frequency = {}
        for element in self.elements:
            elements_frequency[element] = self.get_element_frequency(element)
        return elements_frequency

    def get_normalized_sharq(self):
        elements_normalized_sharq_dict = {}
        elements_freq = self.get_elements_frequency()
        elements_freq = dict(sorted(elements_freq.items(), key=lambda item: item[1], reverse=True))
        self.shap_scores = dict(sorted(self.shap_scores.items(), key=lambda item: item[1], reverse=True))

        for element in self.elements:
            sharq_rank = list(self.shap_scores.keys()).index(element) + 1 # add 1 to not devide by zero
            freq_rank = list(elements_freq.keys()).index(element) + 1
            elements_normalized_sharq_dict[element] = freq_rank/sharq_rank
        elements_normalized_sharq_dict = dict(sorted(elements_normalized_sharq_dict.items(), key=lambda item: item[1], reverse=True))

        return elements_normalized_sharq_dict

    def get_top_bottom(self, frequency_threshold=None, most_frequent_elements_threshold=15):
        if frequency_threshold is not None:
            sharq_scores_filtered = {k: v for k, v in self.shap_scores.items() if self.get_element_frequency(k) >= frequency_threshold}
        else:
            sharq_scores_filtered = self.shap_scores

        # sort sharq scores by elements frequency and get the top 20%
        elements_frequency = self.get_elements_frequency()

        # keep only elements that appear in shap_scores_dict
        elements_frequency = {k: v for k, v in elements_frequency.items() if k in sharq_scores_filtered}

        # sort the dict by frequency
        elements_frequency = dict(sorted(elements_frequency.items(), key=lambda item: item[1], reverse=True))

        # get the most_frequent_elements_threshold % of the elements
        most_frequent_elements_num = int(len(elements_frequency) * most_frequent_elements_threshold / 100)
        most_frequent_elements = dict(itertools.islice(elements_frequency.items(), most_frequent_elements_num))

        # get the sharq scores of the most frequent elements
        most_frequent_elements = {k: v for k, v in sharq_scores_filtered.items() if k in most_frequent_elements.keys()}

        # sort sharq scores
        sorted_sharq_scores = dict(sorted(sharq_scores_filtered.items(), key=lambda item: item[1], reverse=False))

        # get top and bottom 5
        top_5 = dict(itertools.islice(sorted_sharq_scores.items(), 5))
        bottom_5 = dict(itertools.islice(sorted_sharq_scores.items(), len(sorted_sharq_scores) - 5, len(sorted_sharq_scores)))

        # merge the two dicts
        top_bottom = {**top_5, **bottom_5}
        top_bottom = {**top_bottom, **most_frequent_elements}

        # sort the merged dict by sharq score
        top_bottom = dict(sorted(top_bottom.items(), key=lambda item: item[1], reverse=False))
        return top_bottom

