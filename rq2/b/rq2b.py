import argparse
import pandas as pd

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
        node1 for node1 in table_1.node1_class.unique() if "LANE" not in node1.upper()]
    entities_list_2 = [
        node1 for node1 in table_2.node1_class.unique() if "LANE" not in node1.upper()]
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

### Specification 1 ###


def row_calc_1(x, distance):
    val = {
        'truck': {
            'inDFrontOf': 1,
            'inSFrontOf': 2,
        }
    }
    try:
        return val[x['node1_class']][x['edge']] if x[distance] >= 1 and (x['edge'] == 'inDFrontOf' or x['edge'] == 'inSFrontOf') else 0
    except:
        pass


def compute_coverage_spec1(df):
    # Create Result df
    r = df.groupby('sg').size().reset_index(name='counts')
    r.drop(columns=['counts'], inplace=True)

    # ### Filter canvas by specification
    distances = ['near_coll', 'super_near', 'very_near', 'near', 'visible']
    for distance in distances:

        # distance = 'visible'
        # Car in Ego Lane
        q1 = df[(df['node1_class'].isin(['truck']))
                & (df['node2'] == "Ego Lane")]
        # Add 'edge' columns for each entity instance
        q2 = q1[['sg', 'node1', 'node2']].merge(
            df[['sg', 'node1', 'node1_class', 'edge']], how='inner', on=['sg', 'node1'])
        # Filter Car by specified distance
        q3 = q2[q2['edge'] == distance].groupby(
            ['sg', 'node1']).size().reset_index(name=distance)
        # Add 'edge' columns for each entity instance
        q4 = q3.merge(df[['sg', 'node1', 'node1_class', 'edge']],
                      how='inner', on=['sg', 'node1'])
        q4[distance] = q4.apply(lambda x: row_calc_1(x, distance), axis=1)
        q5 = q4[q4[distance] > 0]
        q5 = q5.groupby(['sg', distance]).size().reset_index(name='counts')
        # Merge with Result df
        r = r.merge(q5[['sg', distance]], how='outer', on=['sg'])
        r = r.fillna(0)
    s = r.groupby(distances).size().reset_index(name='counts')
    return s


def print_cov_spec1(s):
    if len(s[(s['near_coll'] == 0) & (s['super_near'] == 0) & (s['very_near'] == 0) & (s['near'] == 0) & (s['visible'] == 0)]) > 0:
        print(f"{len(s)-1} out of 242 ({(len(s)-1) / 242 * 100:.2f}%) | {len(s)-1} ({(len(s)-1) / 242 * 100:.2f}\%)")
    else:
        print(
            f"{len(s)} out of 242 ({len(s) / 242 * 100:.2f}%) | {len(s)} ({(len(s)) / 242 * 100:.2f}\%)")


### Specification 2 ###
def compute_coverage_spec2(df):
    # Create canvas with non-empty Left 1 of Ego Lane
    # Have a Left 1 of Ego Lane
    l1el = df[((df['node1'] == 'Left 1 of Ego Lane'))]['sg']
    canvas = df[df['sg'].isin(l1el)]
    # Non-empty Left 1 of Ego Lane
    ne_l1el = canvas[((canvas['node1_class'].isin(['car'])) & (
        canvas['node2'] == 'Left 1 of Ego Lane'))].sg.unique()
    canvas = canvas[(canvas['sg'].isin(ne_l1el))]

    # Create Result df
    r = df.groupby('sg').size().reset_index(name='counts')
    r.drop(columns=['counts'], inplace=True)

    # Filter canvas by specification
    distances = ['near_coll', 'super_near', 'very_near', 'near', 'visible']
    for distance in distances:
        # Car in Ego Lane
        q1 = canvas[(canvas['node1_class'].isin(['car'])) &
                    (canvas['node2'] == "Left 1 of Ego Lane")]
        # Add 'edge' columns for each entity instance
        q2 = q1[['sg', 'node1', 'node2']].merge(
            canvas[['sg', 'node1', 'node1_class', 'edge']], how='inner', on=['sg', 'node1'])
        # Filter Car by specified distance
        q3 = q2[q2['edge'] == distance].groupby(
            ['sg', 'node1']).size().reset_index(name=distance)
        q3 = q3[q3[distance] > 0]
        q3 = q3.groupby(['sg', distance]).size().reset_index(name='counts')

        # Merge with Result df
        r = r.merge(q3[['sg', distance]], how='outer', on=['sg'])
        r = r.fillna(0)

    s = r.groupby(distances).size().reset_index(name='counts')
    return s


def print_cov_spec2(s):
    if len(s[(s['near_coll'] == 0) & (s['super_near'] == 0) & (s['very_near'] == 0) & (s['near'] == 0) & (s['visible'] == 0)]) > 0:
        print(f"{len(s)-1} out of 31 ({(len(s)-1) / 31 * 100:.2f}%) | {len(s)-1} ({(len(s)-1) / 31 * 100:.2f}\%)")
    else:
        print(
            f"{len(s)} out of 31 ({len(s) / 31 * 100:.2f}%) | {len(s)} ({(len(s)) / 31 * 100:.2f}\%)")


### Specification 3 ###
def row_calc_3(x, distance):
    val = {
        'car': {
            'inDFrontOf': 1,
            'inSFrontOf': 2,
        },
        'truck': {
            'inDFrontOf': 3,
            'inSFrontOf': 4,
        }
    }
    try:
        return val[x['node1_class']][x['edge']] if x[distance] >= 1 and (x['edge'] == 'inDFrontOf' or x['edge'] == 'inSFrontOf') else 0
    except:
        pass


def compute_coverage_spec3(df):
    # Create canvas with nothing on the Left 1 of Ego Lane
    # Have a Left 1 of Ego Lane
    l1el = df[((df['node1'] == 'Left 1 of Ego Lane'))]['sg']
    canvas = df[df['sg'].isin(l1el)]
    # Empty Left 1 of Ego Lane
    vehicle_l1el = canvas[((canvas['node1_class'].isin(['car', 'truck'])) & (
        canvas['node2'] == 'Left 1 of Ego Lane'))].sg.unique()
    canvas = canvas[~(canvas['sg'].isin(vehicle_l1el))]
    canvas

    # Create Result df
    r = df.groupby('sg').size().reset_index(name='counts')
    r.drop(columns=['counts'], inplace=True)

    # Filter canvas by specification
    distances = ['near_coll', 'super_near']
    for distance in distances:
        # Car or Trucks in Ego Lane
        q1 = canvas[(canvas['node1_class'].isin(['car', 'truck']))
                    & (canvas['node2'] == "Ego Lane")]
        # Add 'edge' column for each entity instance
        q2 = q1[['sg', 'node1', 'node2']].merge(
            canvas[['sg', 'node1', 'node1_class', 'edge']], how='inner', on=['sg', 'node1'])
        # Filter Car or Trucks by specified distance
        q3 = q2[q2['edge'] == distance].groupby(
            ['sg', 'node1']).size().reset_index(name=distance)
        # Add 'edge' column for each entity instance
        q4 = q3.merge(canvas[['sg', 'node1', 'node1_class',
                      'edge']], how='inner', on=['sg', 'node1'])
        q4[distance] = q4.apply(lambda x: row_calc_3(x, distance), axis=1)
        q4 = q4[q4[distance] > 0]
        q4 = q4.groupby(['sg', distance]).size().reset_index(name='counts')
        # Merge with Result df
        r = r.merge(q4[['sg', distance]], how='outer', on=['sg'])
        r = r.fillna(0)

    s = r.groupby(distances).size().reset_index(name='counts')
    return s


def print_cov_spec3(s):
    if len(s[(s['near_coll'] == 0) & (s['super_near'] == 0)]) > 0:
        print(f"{len(s)-1} out of 24 ({(len(s)-1) / 24 * 100:.2f}%) | {len(s)-1} ({(len(s)-1) / 24 * 100:.2f}\%)")
    else:
        print(
            f"{len(s)} out of 24 ({len(s) / 24 * 100:.2f}%) | {len(s)} ({(len(s)) / 24 * 100:.2f}\%)")


### Specification 4 ###
def compute_coverage_spec4(df):
    final_lane_list, _, _ = get_full_list(df, df)
    q = df[(df['node1_class'].isin(['car', 'truck'])) & (
        df['edge'] == 'isIn') & (df['node2'].isin(final_lane_list))]
    q = q.groupby(['node1_class', 'node2']).size().reset_index(name='counts')
    return q


def print_cov_spec4(r):
    print(f"{len(r)} out of 22 ({len(r)/22*100:.2f}%) | {len(r)} ({(len(r)) / 22 * 100:.2f}\%)")


def get_lanes(df):
    # All lanes in df
    final_lane_list = [node2 for node2 in df.node2.unique() if "Lane" in node2]
    # Remove Opposing lanes
    final_lane_list = [
        lane for lane in final_lane_list if 'Opposing' not in lane]
    return final_lane_list


def get_canvas(df, lanes):
    final_lane_list = get_lanes(df)
    exclude_lanes = [lane for lane in final_lane_list if lane not in lanes]

    sg_ids = df.sg.unique()
    for lane in lanes:
        q = df[(df['sg'].isin(sg_ids)) & (df['node1'] == lane)]
        q = df[df['sg'].isin(q.sg.unique())]
        exclude_sgs = q[(q['node1'].isin(exclude_lanes))].sg.unique()
        q = q[~q['sg'].isin(exclude_sgs)]
        sg_ids = q.sg.unique()
    return df[df['sg'].isin(sg_ids)]


def get_combinations(df, c, lanes):
    # Create Result df
    r = df.groupby('sg').size().reset_index(name='counts')
    r.drop(columns=['counts'], inplace=True)

    for lane in lanes:
        # Get count of cars and trucks for each sg
        q = c[(c['node1_class'].isin(['car', 'truck'])) & (c['node2'] == lane)].groupby(
            ['sg', 'node1_class', 'node2']).size().reset_index(name='count')
        # Merge with result df
        r = r.merge(q, how='outer', on=['sg']).fillna(0)
        r.drop(columns=['node2', 'count'], inplace=True)
        r.rename(columns={'node1_class': lane}, inplace=True)

    s = r.groupby(lanes).size().reset_index(name='count')
    return s


### Specification 5 ###
def compute_coverage_spec5(df):
    all_cov = []
    all_denominators = []
    results = []
    lane_comb_4 = [
        ['Ego Lane', 'Right 1 of Ego Lane',
            'Right 2 of Ego Lane', 'Right 3 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Ego Lane',
            'Right 1 of Ego Lane', 'Right 2 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Left 2 of Ego Lane',
            'Ego Lane', 'Right 1 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Left 2 of Ego Lane',
            'Left 3 of Ego Lane', 'Ego Lane']
    ]
    lane_comb_3 = [
        ['Ego Lane', 'Right 1 of Ego Lane', 'Right 2 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Ego Lane', 'Right 1 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Left 2 of Ego Lane', 'Ego Lane']
    ]
    lane_comb_2 = [
        ['Ego Lane', 'Right 1 of Ego Lane'],
        ['Left 1 of Ego Lane', 'Ego Lane']
    ]
    lane_comb_1 = [
        ['Ego Lane']
    ]
    all_possible_lane_comb = [lane_comb_1,
                              lane_comb_2, lane_comb_3, lane_comb_4]
    for lane_comb in all_possible_lane_comb:
        print(f"{len(lane_comb)} lane combinations")
        cov = 0
        denominator = len(lane_comb)*3**len(lane_comb)
        for l_lanes in lane_comb:
            # Create Result df
            r = df.groupby('sg').size().reset_index(name='counts')
            r.drop(columns=['counts'], inplace=True)

            # Get canvas
            canvas = get_canvas(df, l_lanes)
            # Non-empty canvas
            if len(canvas) > 0:
                # Get all possible combinations
                combinations = get_combinations(df, canvas, l_lanes)
                # Count combinations
                cov += len(combinations)

        print(
            f"Coverage: {cov} out of {denominator} ({cov/denominator*100:.2f}%)")
        all_cov.append(cov)
        all_denominators.append(denominator)
    print(f"Total Coverage: {sum(all_cov)} out of {sum(all_denominators)} ({sum(all_cov)/sum(all_denominators)*100:.2f}%) | {sum(all_cov)} ({sum(all_cov)/sum(all_denominators)*100:.2f}\%)")


############
### Main ###
############

parser = argparse.ArgumentParser(
    description="RQ2 - A: Difference between test failures and train."
)
parser.add_argument(
    "-csv_folder",
    type=Path,
    required=False,
    default=Path("./rq2/b/carla_csv_orig/"),
    help="Folder containing carla csv files. Default is ./rq2/b/carla_csv_orig/"
)

def main():
    args = parser.parse_args()

    # Load data
    max_vehicles = pd.read_csv(args.csv_folder / 'carla_max_vehicle_v3.csv')
    town01 = pd.read_csv(args.csv_folder / 'carla_max_vehicle_v3_Town01.csv')
    town02 = pd.read_csv(args.csv_folder / 'carla_max_vehicle_v3_Town02.csv')
    town04 = pd.read_csv(args.csv_folder / 'carla_max_vehicle_v3_Town04.csv')
    town10HD = pd.read_csv(args.csv_folder / 'carla_max_vehicle_v3_Town10HD.csv')

    split_list = [max_vehicles, town01, town02, town04, town10HD]
    for split in split_list:
        split['node1_class'] = split.apply(
            lambda x: get_node_class(x.node1), axis=1)
        split['node2_class'] = split.apply(
            lambda x: get_node_class(x.node2), axis=1)

    print("NUMBER OF UNIQUE CLUSTERS:\n")
    print(f"ALL: {len(max_vehicles.sg.unique())}")
    print(f"Town01: {len(town01.sg.unique())}")
    print(f"Town02: {len(town02.sg.unique())}")
    print(f"Town04: {len(town04.sg.unique())}")
    print(f"Town10HD: {len(town10HD.sg.unique())}")
    print("\n---\n")

    ### Specification 1 ###
    print("SPECIFICATION 1:\n")
    print("Town01:")
    s = compute_coverage_spec1(town01)
    print_cov_spec1(s)

    print("Town02:")
    s = compute_coverage_spec1(town02)
    print_cov_spec1(s)

    print("Town04:")
    s = compute_coverage_spec1(town04)
    print_cov_spec1(s)

    print("Town10HD:")
    s = compute_coverage_spec1(town10HD)
    print_cov_spec1(s)

    print("All:")
    s = compute_coverage_spec1(max_vehicles)
    print_cov_spec1(s)


    ### Specification 2 ###
    print("\nSPECIFICATION 2:\n")
    print("Town01:")
    s = compute_coverage_spec2(town01)
    print_cov_spec2(s)

    print("Town02:")
    s = compute_coverage_spec2(town02)
    print_cov_spec2(s)

    print("Town04:")
    s = compute_coverage_spec2(town04)
    print_cov_spec2(s)

    print("Town10HD:")
    s = compute_coverage_spec2(town10HD)
    print_cov_spec2(s)

    print("All:")
    s = compute_coverage_spec2(max_vehicles)
    print_cov_spec2(s)


    ### Specification 3 ###
    print("\nSPECIFICATION 3:\n")
    print("Town01:")
    s = compute_coverage_spec3(town01)
    print_cov_spec3(s)

    print("Town02:")
    s = compute_coverage_spec3(town02)
    print_cov_spec3(s)

    print("Town04:")
    s = compute_coverage_spec3(town04)
    print_cov_spec3(s)

    print("Town10HD:")
    s = compute_coverage_spec3(town10HD)
    print_cov_spec3(s)

    print("All:")
    s = compute_coverage_spec3(max_vehicles)
    print_cov_spec3(s)


    ### Specification 4 ###
    print("\nSPECIFICATION 4:\n")
    print("Town01:")
    s = compute_coverage_spec4(town01)
    print_cov_spec4(s)

    print("Town02:")
    s = compute_coverage_spec4(town02)
    print_cov_spec4(s)

    print("Town04:")
    s = compute_coverage_spec4(town04)
    print_cov_spec4(s)

    print("Town10HD:")
    s = compute_coverage_spec4(town10HD)
    print_cov_spec4(s)

    print("All:")
    s = compute_coverage_spec4(max_vehicles)
    print_cov_spec4(s)


    ### Specification 5 ###
    print("\nSPECIFICATION 5:\n")
    print("Town01:")
    compute_coverage_spec5(town01)

    print("Town02:")
    compute_coverage_spec5(town02)

    print("Town04:")
    compute_coverage_spec5(town04)

    print("Town10HD:")
    compute_coverage_spec5(town10HD)

    print("All:")
    compute_coverage_spec5(max_vehicles)


if __name__ == '__main__':
    main()
