from fileinput import filename
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
import os

def main():
    os.chdir("..")  # Change to the parent directory
    subfolder_path = [x[0] for x in os.walk(os.getcwd() + os.path.sep + 'runs')]
    file_name=[]
    for i in range(1, len(subfolder_path)):
        file_name.append(os.path.basename(subfolder_path[i]))
    print("The files are:\n", file_name)
    
    metrics_name = []
    
    for model_name in file_name:
        in_path = os.getcwd() + os.path.sep + 'runs' + os.path.sep + model_name + os.path.sep
        ex_path = os.getcwd() + os.path.sep + 'score_save'

        # load log data
        parser = argparse.ArgumentParser(description='Export tensorboard data')
        parser.add_argument('--in-path', default=in_path, type=str, required=False, help='Tensorboard event files or a single tensorboard '
                                                                    'file location')
        parser.add_argument('--ex-path', default=ex_path, type=str, required=False, help='location to save the exported data')
        args = parser.parse_args()

        event_data = event_accumulator.EventAccumulator(args.in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        metrics_name = keys.copy()
        # print(keys)
        df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
        for key in tqdm(keys):
            # print(key)
            if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
                df[key] = pd.DataFrame(event_data.Scalars(key)).value
        df.to_csv(args.ex_path + os.path.sep + model_name + '.csv')

    print("Tensorboard data exported successfully")

    print("The metrics are:\n",metrics_name)

if __name__ == '__main__':
    main()

