from .RulesUtils import *
from .DatasetAdjustments import *
import math


def top_1_i_function(df):
    if (type(df) is list and len(df) == 0) or (type(df) is not list and df.shape[0] == 0):
        return 0
    if type(df) is list:
        df = pd.DataFrame(df)
    df = df.sort_values(by=['score'], ascending=False)
    top_df = df.head(1)
    return top_df['score'].sum() * 100


def calc_short_shap_by_all_elements(rules_set, size=None, operator=None, i_func=None, element=None, elements=None):
    bins_rules_dict, grouped_bins_rules_dict, cols_subgroups_dict = get_bins_rules_dict(rules_set, operator, size, element=element, elements=elements)
    if elements is None:
        elements = get_rules_set_elements(rules_set)
    shap_scores = defaultdict(float)
    sub_groups = get_subgroups_from_dict(cols_subgroups_dict)
    bins_scores_dict, mult_scores_dict = get_bins_scores_dict(grouped_bins_rules_dict, len(elements), sub_groups, i_func=i_func)
    if element is not None:
        elements = [element]
    for element in elements:
        if operator == Operators.INFLUENCE:
            influence = calc_element_influence(element, bins_rules_dict)
            shap_scores[element] = influence
            continue
        if operator == Operators.I_TOP:
            i_top = calc_element_i_top(element, bins_rules_dict)
            shap_scores[element] = i_top
            continue
        else:
            element_sub_groups = cols_subgroups_dict[get_elem_col(element)]
        result = calc_presaved_scores_by_all_elements(element, bins_scores_dict, mult_scores_dict, bins_rules_dict, element_sub_groups, operator)
        shap_scores[element] = result
    shap_scores = sorted(shap_scores.items(), key=lambda x:x[1])
    return shap_scores


def get_subgroups_from_dict(elements_subgroups_dict):
    sub_groups = []
    for key, value in elements_subgroups_dict.items():
        groups = [x for x in value]
        sub_groups.extend(groups)
    # remove duplicates
    sub_groups = list(set(sub_groups))
    # convert string of list to list
    sub_groups = [ast.literal_eval(group) for group in sub_groups]
    return sub_groups


def get_bins_scores_dict(grouped_bins_rules_dict, tot_elem_num, sub_groups=None, i_func=top_1_i_function):
    temp_dict = grouped_bins_rules_dict.copy()
    bins_scores_dict = defaultdict(float)
    mult_scores_dict = defaultdict(float)
    if sub_groups is not None and len(grouped_bins_rules_dict) > len(sub_groups) > 0:
        filtered_dict = defaultdict(str)
        for group in sub_groups:
            group.sort()
            filtered_dict[str(group)] = temp_dict[str(group)]
        temp_dict = filtered_dict
    for key, value in temp_dict.items():
        if value is None or len(value) == 0:
            bins_scores_dict[key] = 0
            continue
        sub_group_len = key.count('_')
        if mult_scores_dict[sub_group_len] == 0:
            if tot_elem_num == sub_group_len:
                sub_group_len = sub_group_len - 1
            if sub_group_len > tot_elem_num:
                mult_scores_dict[sub_group_len] = 0
            else:
                mult_scores_dict[sub_group_len] = (sterling(sub_group_len) / sterling(tot_elem_num)) * sterling(tot_elem_num - sub_group_len - 1)
        bins_scores_dict[key] = i_func(pd.DataFrame(value))

    if tot_elem_num >= 2:
        mult_scores_dict[2] = (math.factorial(2) * math.factorial(
                tot_elem_num - 1 - 2)) / math.factorial(tot_elem_num)
    mult_scores_dict[1] = (math.factorial(1) * math.factorial(
            tot_elem_num - 1 - 1)) / math.factorial(tot_elem_num)
    return bins_scores_dict, mult_scores_dict


def sterling(n):
    if n == 1:
        return 1

    # value of natural e
    e = 2.71

    # evaluating factorial using stirling approximation
    z = (math.sqrt(2 * 3.14 * n) * math.pow((n / e), n))
    return math.floor(z)


def calc_element_influence(element, bins_rules_dict):
    rules_with_element = bins_rules_dict[element]
    rules_with_element = pd.DataFrame(rules_with_element)
    return rules_with_element['score'].sum()


def calc_element_i_top(element, bins_rules_dict):
    rules_with_element = bins_rules_dict[element]
    rules_with_element = pd.DataFrame(rules_with_element)
    rules_with_element = rules_with_element.sort_values(by=['score'], ascending=False)
    return rules_with_element['score'].head(1).sum()


def calc_presaved_scores_by_all_elements(element, bins_scores_dict, mult_scores_dict, bins_rules_dict,
                                         filtered_sub_groups, operator=None):
    sigma = 0
    if operator == Operators.INFLUENCE:
        influence = calc_element_influence(element, bins_rules_dict)
        return influence, []
    elif operator == Operators.I_TOP:
        i_top = calc_element_i_top(element, bins_rules_dict)
        return i_top, []
    else:
        sub_groups = filtered_sub_groups

    for sub_group in sub_groups:
        sharq_score = shapley_precalc_inner_sigma(element, sub_group, bins_scores_dict, mult_scores_dict)
        sigma = sigma + sharq_score
    return sigma


def shapley_precalc_inner_sigma(element, sub_group, bins_scores_dict, mult_scores_dict):
    if type(sub_group) == str:
        sub_group = ast.literal_eval(sub_group)
    sub_group_with_element = list(sub_group)
    sub_group_with_element.append(element)
    multiplier = bins_scores_dict[str(sorted(sub_group_with_element))] - bins_scores_dict[str(sorted(sub_group))]
    shapley_score = mult_scores_dict[len(sub_group)] * multiplier
    return shapley_score

