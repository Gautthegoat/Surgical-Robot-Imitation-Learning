from source.datasets.components import Transform

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class DatasetIL(Dataset):
    def __init__(self, config, archive_folder_path, logger, mode='train', norm_stats=None):
        self.config = config
        self.archive_folder_path = archive_folder_path
        self.logger = logger
        self.mode = mode
        self.norm_stats = norm_stats

        self.nb_futur_logs = config['data']['length_actions']
        self.height = config['data']['frame_size'][0]
        self.width = config['data']['frame_size'][1]
        self.type_norm = config['data']['norm']
        self.transform = Transform(config)
        
        if mode == 'train':
            self.data_dir = config['data']['train_data_path']
        elif mode == 'val':
            self.data_dir = config['data']['val_data_path']

        self.data = self.create_data()
        self.data, self.norm_stats = self.apply_norm(self.data)
        self.data_padded = self.pad_data()
        self.plot_distr()

    def pad_data(self):
        """
        Pad the data so that we add nb_futur_logs frames to the end of each demo as the same as the last frame
        """
        self.logger.info("Padding the data")
        data_padded = []
        for demo in self.data:
            last_frame = demo[-1]
            padded_demo = np.repeat(last_frame[np.newaxis, :], self.nb_futur_logs, axis=0)
            data_padded.append(np.concatenate((demo, padded_demo), axis=0))
        data_padded = np.array(data_padded, dtype=object)
        return data_padded

    def plot_distr(self):
        """
        Save Matplotlib plots of the distribution of the basic and type logs
        """
        self.logger.info("Plotting the distribution of the logs")
        import matplotlib.pyplot as plt
        folder_path = f'{self.archive_folder_path}/plots/{self.mode}_distr'
        os.makedirs(folder_path, exist_ok=True)
        logs_array = np.array([]).reshape(0, 12)
        for demo in self.data:
            logs_array = np.concatenate((logs_array, demo[:, 1:]), axis=0)
        data_array = logs_array.astype(np.float32)
        for i in range(data_array.shape[1]):
            plt.hist(data_array[:, i], bins=100, alpha=0.5, label='Distribution')
            plt.legend(loc='upper right')
            plt.title(f'Joint {i+1}')
            plt.savefig(f'{folder_path}/joint_{i+1}.png')
            plt.clf()

    def apply_norm(self, data_list):
        """
        Ǹormalize the data_list logs
        """
        self.logger.info("Normalizing the data")
        # Get in a numpy array all the logs
        logs_array = np.array([]).reshape(0, 12)
        for demo in data_list:
            logs_array = np.concatenate((logs_array, demo[:, 1:]), axis=0)
        logs_array = logs_array.astype(np.float32)

        if self.type_norm == 'standard':
            # normalize to mean 0 and std 1
            if self.norm_stats is None:
                means = np.mean(logs_array, axis=0)
                stds = np.std(logs_array, axis=0)
            else:
                means, stds = self.norm_stats
            normalized_data = (logs_array - means) / stds
            stats = [means, stds]
        elif self.type_norm == 'minmax':
            # normalize to [0, 1]
            eps = 1e-8
            if self.norm_stats is None:
                mins = np.min(logs_array, axis=0)
                maxs = np.max(logs_array, axis=0)
            else:
                mins, maxs = self.norm_stats
            normalized_data = (logs_array - mins) / ((maxs - mins)+eps)
            stats = [mins, maxs]

        # Replace the logs in the data_list
        count = 0
        for i, demo in enumerate(data_list):
            demo[:, 1:] = normalized_data[count:count+len(demo), :]
            count += len(demo)

        return data_list, stats

    def create_data(self):
        """
        From the data_dir, create a list containing a np.array for each demo -> [[[frame_path, j1,j2,...], ... ], ...] shape (nb_demos, nb_frames, 12+1)
        """
        self.logger.info(f"Extracting data from {self.data_dir}")
        demo_count = 0
        data = []
        demos_folder = [folder for folder in os.listdir(self.data_dir) if folder.endswith("demos")]
        for demo_folder in demos_folder:
            demos_path = os.listdir(os.path.join(self.data_dir, demo_folder))
            demos_path = [demo for demo in demos_path if demo.startswith("demo")]
            for demo in demos_path:
                df = pd.read_csv(os.path.join(self.data_dir, demo_folder, demo, "index.csv"))
                demo_list = []
                for index, row in df.iterrows():
                    frame_path = os.path.join(self.data_dir, demo_folder, demo, row['frame'])
                    actions = ast.literal_eval(row['logs'])
                    actions.pop('camera_north')
                    actions.pop('camera_east')
                    combined = [frame_path] + list(actions.values())
                    demo_list.append(combined)
                demo_array = np.array(demo_list, dtype=object)
                demo_count -=- 1 # Don't you dare change this line, gucci.
                data.append(demo_array)
        data = np.array(data, dtype=object)
        return data 

    def _get_demo_and_index(self, index):
        """
        Helper method to find the demo index and number index corresponding to the flattened index
        """
        count = 0
        for i, demo in enumerate(self.data):
            if index < count + len(demo):
                return i, index - count
            count += len(demo)
        raise IndexError("Index out of range")

    def __len__(self):
        return sum([len(demo) for demo in self.data])
    
    def __getitem__(self, idx):
        """
        current_frame: torch.Tensor of shape (c, h, w)
        actions: torch.Tensor of shape (nb_futur_logs, joint_dim)
        """
        # Get the demo index and the frame index in the demo
        demo_idx, frame_idx = self._get_demo_and_index(idx)
        frame_path, actions = self.data_padded[demo_idx][frame_idx, 0], self.data_padded[demo_idx][frame_idx:frame_idx+self.nb_futur_logs+1, 1:]
        frame_array = np.load(frame_path)
        frame = self.transform.transform_frame(self.height, self.width, frame_array)
        actions = actions.astype(np.float32)
        actions = self.transform.transform_actions(actions)

        return (frame, actions[0], actions[1:]), actions[1:] # (current_frame, current_qpos, future_actions), future_actions