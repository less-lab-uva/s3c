import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from pathlib import Path


########################
### Common Functions ###
########################

def get_node_class(node):
    if "LANE" in node.upper():
        return "lane"
    elif "ROOT ROAD" in node.upper():
        return "root"
    elif "EGO" in node.upper():
        return "ego"
    else:
        return f"{node[:node.rfind('_')]}"


def get_full_list(table_1, table_2):
    # table_2 is only used to define a common set of lanes with table_1
    lane_list_1 = [node2 for node2 in table_1.node2.unique()
                   if "Lane" in node2]
    lane_list_2 = [node2 for node2 in table_2.node2.unique()
                   if "Lane" in node2]
    final_lane_list = lane_list_2.copy()
    for lane_name in lane_list_1:
        if lane_name not in lane_list_2:
            final_lane_list.append(lane_name)

    final_lane_list = sorted(final_lane_list)

    # table_2 is only used to define a common set of entities with table_1
    entities_list_1 = [
        node1 for node1 in table_1.node1_class.unique() if "Lane" not in node1]
    entities_list_2 = [
        node1 for node1 in table_2.node1_class.unique() if "Lane" not in node1]
    final_entities_list = entities_list_2.copy()
    for entity_name in entities_list_1:
        if entity_name not in entities_list_2:
            final_entities_list.append(entity_name)

    final_entities_list = sorted(final_entities_list)

    # table_2 is only used to define a common set of relationships with table_1
    relationship_list_1 = [
        edge for edge in table_1.edge.unique() if "isIn" not in edge]
    relationship_list_2 = [
        edge for edge in table_2.edge.unique() if "isIn" not in edge]
    final_relationship_list = relationship_list_2.copy()
    for entity_name in relationship_list_1:
        if entity_name not in relationship_list_2:
            final_relationship_list.append(entity_name)

    final_relationship_list = sorted(final_relationship_list)

    return final_lane_list, final_entities_list, final_relationship_list


def get_entity_times_lane(df):
    """
    df filter:
        - node1_class != 'lane': node1 is an entity
        - node2_class == 'lane': node2 is a lane
    group by:
        - sg: scene graph
        - node1: entity instance
        - node2: lane instance
    """
    return df[(df['node1_class'] != 'lane') & (df['node2_class'] == "lane")].groupby(['sg', 'node1', 'node2']).size().reset_index(name='count')


def get_entity_times_relationship(df):
    """
    df filter:
        - node1_class != 'lane': node1 is an entity
        - AND [
            edge != 'isIn': edge is a relationship between (ego and entity) OR
            node1_class == 'traffic_light': node1 is a traffic light because it has 'isIn' relationship
        ]
    group by:
        - sg: scene graph
        - node1: entity instance
        - edge: relationship
    """
    return df[(df['node1_class'] != 'lane') & ((df['edge'] != "isIn") | (df['node1_class'] == "traffic_light"))].groupby(['sg', 'node1', 'edge']).size().reset_index(name='count')


def join_entity_lane_relationship(df_el, df_er):
    """
    df_el: entity times lane
    df_er: entity times relationship
    """
    q = df_el.merge(df_er, how='inner', on=['sg', 'node1'])
    q['node1_class'] = q.apply(lambda x: get_node_class(x.node1), axis=1)
    q = q.groupby(['sg', 'node1_class', 'node2', 'edge']).sum().reset_index()
    q.drop(columns=['count_x'], inplace=True)
    q.rename(columns={'count_y': 'count'}, inplace=True)
    return q


def create_entity_lane_relationship_table(df):
    q = get_entity_times_lane(df)
    q2 = get_entity_times_relationship(df)
    return join_entity_lane_relationship(q, q2)


def create_full_empty_table(table_1, table_2):
    final_lane_list, final_entities_list, final_relationship_list = get_full_list(
        table_1, table_2)
    all_combinations = itertools.product(table_1.sg.unique(
    ), final_entities_list, final_lane_list, final_relationship_list)
    dict_list = []
    for sg, entity, lane, relationship in all_combinations:
        # Append dictionary to list
        dict_list.append({'sg': sg, 'node1_class': entity,
                         'node2': lane, 'edge': relationship, 'count': 0})
    # Create Pandas DataFrame from list of dictionaries
    q_all = pd.DataFrame.from_dict(dict_list)
    return q_all


def complete_table(table_1, table_2):
    q_all = create_full_empty_table(table_1, table_2)
    q_all = q_all.merge(table_1, how='left', on=[
                        'sg', 'node1_class', 'node2', 'edge']).fillna(0)
    q_all.drop(columns=['count_x'], inplace=True)
    q_all.rename(columns={'count_y': 'count'}, inplace=True)
    return q_all


def get_data_for_training(ct_1, label_1, ct_2, label_2):
    # Pivot the table from having one row per combination, to have one column per combinations
    p1 = ct_1.pivot(columns=['node1_class', 'node2', 'edge'], index='sg', values='count').droplevel(
        0, axis=1).droplevel(0, axis=1).reset_index().rename_axis(None, axis=1)
    p2 = ct_2.pivot(columns=['node1_class', 'node2', 'edge'], index='sg', values='count').droplevel(
        0, axis=1).droplevel(0, axis=1).reset_index().rename_axis(None, axis=1)
    # Use all columns except the first one (sg) as features to create the embedding
    d = np.concatenate([p1.iloc[:, 1:].values, p2.iloc[:, 1:].values], axis=0)
    sgs = p1.iloc[:, 0].append(p2.iloc[:, 0])
    l = [label_1]*len(p1)
    l.extend([label_2]*len(p2))
    return d, l, sgs


############
### Main ###
############

parser = argparse.ArgumentParser(
    description="RQ2 - A: Difference between test failures and train."
)
parser.add_argument(
    "-cluster_path_1",
    type=Path,
    required=False,
    default=Path("./rq2/a/splits_csv_orig/test_fail_diff_train.csv"),
    help="Test fail diff train Cluster (.json file). Default is ./rq2/a/splits_csv_orig/test_fail_diff_train.csv"
)
parser.add_argument(
    "-cluster_path_2",
    type=Path,
    required=False,
    default=Path("./rq2/a/splits_csv_orig/train_diff_test_fail.csv"),
    help="Train diff test fail Cluster (.json file). Default is ./rq2/a/splits_csv_orig/train_diff_test_fail.csv"
)

def main():
    args = parser.parse_args()

    df_1 = pd.read_csv(args.cluster_path_1)
    df_1['node1_class'] = df_1.apply(lambda x: get_node_class(x.node1), axis=1)
    df_1['node2_class'] = df_1.apply(lambda x: get_node_class(x.node2), axis=1)

    df_2 = pd.read_csv(args.cluster_path_2)
    df_2['node1_class'] = df_2.apply(lambda x: get_node_class(x.node1), axis=1)
    df_2['node2_class'] = df_2.apply(lambda x: get_node_class(x.node2), axis=1)

    table_1 = create_entity_lane_relationship_table(df_1)
    table_2 = create_entity_lane_relationship_table(df_2)

    ct_1 = complete_table(table_1, table_2)
    ct_2 = complete_table(table_2, table_1)

    final_lane_list, final_entities_list, final_relationship_list = get_full_list(
        table_1, table_2)
    combinations = itertools.product(
        final_entities_list, final_lane_list, final_relationship_list)
    map_list = []
    for entity, lane, relationship in combinations:
        map_list.append(f"{entity} - {lane} - {relationship}")

    ### Create Decision Tree ###
    # 0 -> test_fail
    # 1 -> train
    data, labels, sgs = get_data_for_training(ct_1, 0, ct_2, 1)

    # Train decision Tree
    clf = DecisionTreeClassifier(random_state=0, max_depth=27)
    clf.fit(data, labels)

    # Decision Tree statistics
    preds = clf.predict(data)
    c = 0
    for idx, p in enumerate(preds):
        if p == 0 and p == labels[idx]:
            c += 1

    features = clf.tree_.feature
    print("DECISION TREE STATISTICS:\n")
    print(
        f"Correct failure prediction: {c} of {(np.array(labels) == 0).sum()}")
    print(f"Accuracy score: {accuracy_score(labels, preds)}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    print(f"Max depth: {clf.get_depth()}")
    print(f"{len(features)} number of features")

    # Calculate average depth of the tree for failures
    all_depths = []
    f = [idx for idx in range(len(labels)) if labels[idx] == 0]
    partial_pred = clf.predict(data[:len(f)])
    for embedding, label, pred in zip(data[:len(f)], labels[:len(f)], partial_pred):
        if label == 0 and pred == 0:
            decision_path = clf.decision_path(embedding.reshape(1, -1))
            depth = len(decision_path.indices) - 1
            all_depths.append(depth)
    print(f"Average depth: {sum(all_depths)/len(all_depths)}")
    print("\n --- \n")

    # Small plot
    plt.figure(figsize=(15, 15))
    tree.plot_tree(clf)
    plt.savefig('./study_data/figures/tree_small.png')
    plt.close()

    # Big plot
    plt.figure(figsize=(300, 300))
    tree.plot_tree(clf)
    plt.savefig('./study_data/figures/tree.png')
    plt.close()

    # Predicates explanations
    print("PREDICATES EXPLANATIONS:\n")
    print(f"52: {map_list[52]}")
    print(f"\t129: {map_list[129]}")
    print()
    print(f"52: {map_list[52]}")
    print(f"\t455: {map_list[455]}")
    print(f"\t\t7: {map_list[7]}")
    print(f"\t\t\t101: {map_list[101]}")


if __name__ == '__main__':
    main()
