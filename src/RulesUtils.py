import ast
from enum import Enum
from efficient_apriori import apriori as apriori_algo
import math
from collections import defaultdict
from scipy.special import binom
import numpy as np
import pandas as pd


SELECTED_COLUMNS = ['score', 'lift', 'support', 'confidence', 'lhs', 'rhs']


class Operators(Enum):
    INFLUENCE = 7
    I_TOP = 11
    OPT_MIN = 19
    KERNEL_SHAP_RANDOM = 23


def create_transactions(df, col_to_separate=None, algo='apriori'):
    if col_to_separate:
        neg_df = df[df[col_to_separate] == 0].drop([col_to_separate], axis=1)
        pos_df = df[df[col_to_separate] == 1].drop([col_to_separate], axis=1)

        neg_df_clean = comp_notnull1(neg_df)
        pos_df_clean = comp_notnull1(pos_df)

        transactions_neg = []
        for r in neg_df_clean:
            transactions_neg.append(list(r.items()))

        transactions_pos = []
        for r in pos_df_clean:
            transactions_pos.append(list(r.items()))

        return transactions_neg, transactions_pos
    else:
        if algo == 'apriori':
            df_clean = comp_notnull1(df)
            transactions = []
            for r in df_clean:
                transactions.append(list(r.items()))
        else:
            transactions = comp_list(df)
        return transactions


def comp_notnull1(df1):
    return [{k: v for k, v in m.items() if pd.notnull(v)} for m in df1.to_dict(orient='records')]


def comp_list(df1):
    dict_list = [{k: v for k, v in m.items() if pd.notnull(v)} for m in df1.to_dict(orient='records')]
    tran_list = []
    idx=0
    for d in dict_list:
        d = {k: v for k, v in d.items() if v == 1}
        tran_list.append(list(d.keys()))
        idx = idx+1
    return tran_list


def is_score_function(rule):
    return math.sqrt(rule['support'] * rule['lift'])


def create_rules_from_transactions(transactions, lift_threshold=None, score_func=is_score_function, min_support=0.05):
    positive_t = []
    for t in transactions:
        t = [a for a in t if not a[1] == 0]
        positive_t.append(t)
    itemsets, rules = apriori_algo(positive_t, min_support=min_support, output_transaction_ids=True)
    if len(rules) == 0:
        return rules
    attrs = [a for a in dir(rules[0]) if not a.startswith("_")]
    rules_rec = []
    for r in rules:
        rdict = {}
        for a in attrs:
            rdict[a] = getattr(r, a)
            rdict["rule"] = str(r).split("} (")[0] + "}"
            rdict["len_l"] = len(r.lhs)
            rdict["len_r"] = len(r.rhs)
            rdict["num_of_elements"] = len(r.lhs) + len(r.rhs)
        rules_rec.append(rdict)

    rules_set = pd.DataFrame(rules_rec)
    rules_set.set_index('rule', inplace=True)
    if score_func is None:
        rules_set['score'] = rules_set.apply(lambda row: is_score_function(row), axis=1)
    else:
        rules_set['score'] = rules_set.apply(lambda row: score_func(row), axis=1)
    rules_set = rules_set[SELECTED_COLUMNS]
    if lift_threshold is not None:
        rules_set = rules_set[(rules_set['lift'] >= 1 + lift_threshold)]
        # rules_set = rules_set[(rules_set['lift'] <= 1-lift_threshold) | (rules_set['lift'] >= 1+lift_threshold)] # you can use this line if you want to use the lift threshold from both sides of '1'
    rules_set.index = range(len(rules_set))
    rules_set = beautify_rhs_lhs(rules_set)
    return rules_set


def beautify_rhs_lhs(rules_set):
    for i, rule in rules_set.iterrows():
        lhs_elements = [element[0] for element in rule['lhs']]
        rhs_elements = [element[0] for element in rule['rhs']]
        rules_set.at[i, 'lhs'] = lhs_elements
        rules_set.at[i, 'rhs'] = rhs_elements
    return rules_set


def get_rules_set_elements(rules_set):
    elements = []
    for i, rule in rules_set.iterrows():
        elements.extend(get_rules_elements(rule))
    elements = list(set(elements))
    if type(elements[0]) == tuple:
        elements = [element[0]+'_'+element[1] for element in elements]
    return elements


def get_rules_elements(rule):
    if type(rule) == tuple:
        rule = rule[1]
    if type(rule['rhs']) == list:
        elements = rule['rhs'] + rule['lhs']
    elif type(rule['rhs']) == pd.Series:
        elements = ast.literal_eval(rule['rhs'].iloc[0]) + ast.literal_eval(rule['lhs'].iloc[0])
    else:
        elements = [element for element in ast.literal_eval(rule['rhs'].replace('\n', '').replace('\' \'', '\',\'')) + ast.literal_eval(rule['lhs'].replace('\n', '').replace('\' \'', '\',\''))] # hadar
        elements = sorted(elements)

    if type(elements[0]) == tuple:
        elements = [element[0]+'_'+element[1] for element in elements]
    return elements


def add_elements_columns(rules_set):
    rules_set['elements'] = rules_set.apply(lambda row: sorted(get_rules_elements(row)), axis=1)
    rules_set['num_of_elements'] = rules_set.apply(lambda row: len(get_rules_elements(row)), axis=1)
    return rules_set


def get_bins_rules_dict(rules_set, operator=Operators.OPT_MIN, size=None, elements=None, element=None):
    if elements is not None:
        all_elements = elements
    else:
        all_elements = get_rules_set_elements(rules_set)
    bins_rules_dict = defaultdict(list)
    grouped_bins_rules_dict = defaultdict(list)
    cols_subgroups_dict_set = defaultdict(set)
    cols_subgroups_dict = defaultdict(list)

    if operator == Operators.KERNEL_SHAP_RANDOM:
        all_columns = set([e.split('_')[0] for e in all_elements])
        add_rules_set_kernel_shap_prob_column(rules_set, len(all_columns))
        rules_set = rules_set.sample(frac=size/100, weights=rules_set["kernel_shap_prob"])

    for i, rule in rules_set.iterrows():
        rule_elements = rule['elements']

        grouped_bins_rules_dict[str(rule_elements)] = [rule]
        if element is None or (element is not None and element in rule_elements):
            for e in rule_elements:
                bins_rules_dict[e].append(rule)
        if operator in [Operators.INFLUENCE, Operators.I_TOP]:
            continue
        add_to_cols_subgroups_dict(cols_subgroups_dict_set, all_elements, rule_elements)
    if element is not None:
        elem_cols = [element.split('_')[0]]
    else:
        elem_cols = get_cols_from_elements(all_elements)
    for c in elem_cols:
        cols_subgroups_dict[c] = list(cols_subgroups_dict_set[c])
    return bins_rules_dict, grouped_bins_rules_dict, cols_subgroups_dict


def get_subgroup_prob_by_size(size, p):
    return (p-1)/(binom(p, size)*(size*(p-size))) * 1000


def add_rules_set_kernel_shap_prob_column(rules_set, p):
    rules_set['kernel_shap_prob'] = rules_set.apply(lambda row: get_subgroup_prob_by_size(len(get_rules_elements(row)), p), axis=1)
    rules_set['kernel_shap_prob'] = rules_set['kernel_shap_prob'].fillna(0)
    rules_set['kernel_shap_prob'] = rules_set['kernel_shap_prob'].replace([np.inf, -np.inf], 0)
    return rules_set


def add_to_cols_subgroups_dict(cols_subgroups_dict, all_elements, rule_elements, element=None):
    if element is not None:
        cols = element.split('_')[0]
    else:
        cols = get_cols_from_elements(all_elements)
    rule_cols = get_cols_from_elements(rule_elements)
    if element is not None:
        if cols not in rule_cols:
            initialize_cols_subgroups_dict(cols_subgroups_dict, cols, rule_elements)
    else:
        for c in cols:
            if c not in rule_cols:
                initialize_cols_subgroups_dict(cols_subgroups_dict, c, rule_elements)

    for e in rule_elements:
        sorted_partial, partial_cols = get_sorted_partial_partial_cols(rule_elements, e)
        if element is not None:
            if cols not in partial_cols:
                initialize_cols_subgroups_dict(cols_subgroups_dict, cols, sorted_partial)
        else:
            for c in cols:
                if c not in partial_cols:
                    initialize_cols_subgroups_dict(cols_subgroups_dict, c, sorted_partial)


def get_cols_from_elements(elements):
    cols = [get_elem_col(element) for element in elements]
    cols = list(set(cols))
    return cols


def get_elem_col(element):
    return element.split('_')[0]


def initialize_cols_subgroups_dict(cols_subgroups_dict, c, rule_elements):
    cols_subgroups_dict[c].add(str(rule_elements))


def get_sorted_partial_partial_cols(rule_elements, e):
    partial_elements = rule_elements.copy()
    partial_elements.remove(e)
    sorted_partial = str(partial_elements)
    partial_cols = get_cols_from_elements(partial_elements)
    return sorted_partial, partial_cols

