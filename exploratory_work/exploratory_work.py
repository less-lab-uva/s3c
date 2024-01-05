import os
import json
import pandas as pd


########################
### Common Functions ###
########################


def get_node_class(node):
    if "Lane" in node:
        return "lane"
    elif "Root Road" in node:
        return "root"
    elif "ego" in node:
        return "ego"
    else:
        return f"{node[:node.rfind('_')]}"


def get_distribution_of(df, entity):
    # Get all scene graphs with number of images
    sgs_df = df.groupby(['sg', 'n_img']).size().reset_index(name='counts')
    # Repeat each scene graph n_img times
    full_df = sgs_df.reindex(sgs_df.index.repeat(sgs_df.n_img))
    # Remove n_img and counts column
    full_df.drop(columns=['n_img', 'counts'], inplace=True)
    # Get all SGs with entity
    entity_df = df[(df["node1_class"] == entity) & (df["edge"] == "isIn")].groupby(
        ['sg']).size().reset_index(name='counts')
    # Merge both dataframes to get SGs with entity and SGs without entity
    full_entity_df = full_df.merge(entity_df, how='outer', on='sg').fillna(0)
    # Get mean and std of entity over number of images
    return full_entity_df['counts'].mean(), full_entity_df['counts'].std()


############
### Main ###
############

def main():
    # Load csv files
    cityscapes_df = pd.read_csv("./dataset_csv/Cityscapes.csv")
    udacity_df = pd.read_csv("./dataset_csv/Udacity.csv")
    nuscenes_df = pd.read_csv("./dataset_csv/Nuscenes.csv")
    sully_df = pd.read_csv("./dataset_csv/Sully.csv")
    commaai_df = pd.read_csv("./dataset_csv/CommaAi.csv")

    dataset_list = [cityscapes_df, udacity_df,
                    nuscenes_df, sully_df, commaai_df]
    dataset_names = ["Cityscapes", "Udacity", "Nuscenes", "Sully", "CommaAi"]

    for df in dataset_list:
        df['node1_class'] = df.apply(lambda x: get_node_class(x.node1), axis=1)
        df['node2_class'] = df.apply(lambda x: get_node_class(x.node2), axis=1)

    table = pd.DataFrame(
        columns=['dataset', 'person', 'bicycle', 'motorcycle', 'bus', 'car'])

    for df, d_name in zip(dataset_list, dataset_names):
        row = {}
        for entity in table.columns[1:]:
            row[entity] = get_distribution_of(df, entity)
        table = table.append({'dataset': d_name,
                              'person': f"{row['person'][0]:.2f} $\pm$ {row['person'][1]:.2f} &",
                              'bicycle': f"{row['bicycle'][0]:.2f} $\pm$ {row['bicycle'][1]:.2f} &",
                              'motorcycle': f"{row['motorcycle'][0]:.2f} $\pm$ {row['motorcycle'][1]:.2f} &",
                              'bus': f"{row['bus'][0]:.2f} $\pm$ {row['bus'][1]:.2f} &",
                              'car': f"{row['car'][0]:.2f} $\pm$ {row['car'][1]:.2f}"
                              }, ignore_index=True)
    print('=== Right Half of Table 3 ===')
    print(table)

    if os.path.exists(".clusters_unpacked"):
        dataset_names = ['CommaAi', 'Nuscenes', 'Sully', 'Udacity', 'Cityscapes', 'Union']
        datasets = {}
        for dataset_name in dataset_names:
            with open(f'clusters/{dataset_name}.json') as f:
                datasets[dataset_name] = json.load(f)

        union = datasets['Union']
        all_images = set()
        union_map = {}
        for dataset_name in dataset_names:
            if dataset_name == 'Union':
                continue
            all_images.update(datasets[dataset_name]['image_files'])
            union_map.update({image: dataset_name for image in datasets[dataset_name]['image_files']})
        asg_table = pd.DataFrame(columns=['dataset', 'Images', 'Images %', '$|ASG_C|$', '$|ASG_C| %$', '$|Img|/|ASG_C|$'])
        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]
            asg_table = asg_table.append({'dataset': dataset_name,
                                          'Images': f"{len(dataset['image_files'])} &",
                                          'Images %': f"{100*len(dataset['image_files']) / len(union['image_files']):0.0f}\\% &",
                                          '$|ASG_C|$': f"{len(dataset['clusters'])} &",
                                          '$|ASG_C| %$': f"{100*len(dataset['clusters']) / len(union['clusters']):0.0f}\\% &",
                                          '$|Img|/|ASG_C|$': f"{len(dataset['image_files']) / len(dataset['clusters']):0.2f}"
                                          }, ignore_index=True)
        print()
        print('=== Left Half of Table 3 ===')
        print(asg_table)
    else:
        print('It appears that the cluster data has not been unpacked.')
        print('Please run unpack_exploratory_work.sh and then re-run this script.')


if __name__ == '__main__':
    main()
