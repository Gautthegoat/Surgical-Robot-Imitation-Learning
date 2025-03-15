import av
import numpy as np
from PIL import Image
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import sqlite3
from colorama import Fore, init
import cv2
import math
import readline

class Extract_Demos:
    def __init__(self, video_path, verbose=False):
        self.verbose = verbose
        self.video_path = os.path.abspath(video_path)
        name_video = os.path.basename(self.video_path).split('.')[0]
        self.demos_path = os.path.dirname(self.video_path) + '/' + name_video + '_demos'
        os.makedirs(self.demos_path, exist_ok=True)
        self.logs_path = os.path.dirname(self.video_path) + '/' + name_video + '_logs'
        print(f"{Fore.MAGENTA} [info]{Fore.RESET} Video path: {self.video_path}")
        print(f"{Fore.MAGENTA} [info]{Fore.RESET} Saving demos in {self.demos_path}")
        self.df_timestamps = pd.DataFrame(columns=['timestamp', 'frame'])
        # self.fps = self.get_fps(video_path)
        # print(f"{Fore.RED} [fps extractor info]{Fore.RESET} Video FPS: {self.fps}")
        
        # For now we do one at a time
        self.count_frame = 0
        self.prev_second = None
        # self.create_video()
        # self.extract_demos()
        # self.add_logs()

        # # Function to transform a demo into a video
        demo_number = 30
        self.watch_demo(demo_number)

    def bcd_to_dec(self, bcd):
        low_nibble = bcd & 0x0F
        high_nibble = (bcd >> 4) & 0x0F
        return low_nibble + 10*high_nibble

    def decode_s12m_timecode(self, side_data):
        raw_data = bytes(side_data)
        # Extract timecode components
        h = self.bcd_to_dec(raw_data[4] & 0x7F)
        m = self.bcd_to_dec(raw_data[5] & 0x7F)
        s = self.bcd_to_dec(raw_data[6] & 0x7F)
        if s != self.prev_second:
            self.prev_second = s
            self.count_frame = 0
        ms = (self.count_frame / math.ceil(self.fps) + 1/(2*math.ceil(self.fps))) * 1000  # ms = (frame_number / fps + 1/(2*fps)) * 1000
        # print(f"{Fore.CYAN} [DEBUG]{Fore.RESET} h: {h} | m: {m} | s: {s} | ms: {ms}")
        self.count_frame += 1
        return h, m, s, ms
    
    def to_iso8601(self, h, m, s, ms):
        date_string = os.path.splitext(os.path.basename(self.video_path))[0]
        date_object = datetime.strptime(date_string, "%Y-%m-%d-%H-%M-%S")
        date_object = date_object.replace(hour=h, minute=m, second=s, microsecond=int(ms)*1000, tzinfo=timezone.utc)
        return date_object.isoformat()
    
    def extract_timestamps(self, frame):
        for side_data in frame.side_data:
            if side_data.type == 'S12M_TIMECODE':
                return self.decode_s12m_timecode(side_data)
            else:
                return None
            
    def editable_input(self, prompt, prefill=''):
        """
        Display a prompt with prefilled text that the user can edit.
        """
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook()
    
    def get_start_end(self, nb_demo, last_end):
        """
        Gather the start and end timestamps of each demonstration
        """
        print(f'\n{Fore.YELLOW}[Demo {nb_demo}]{Fore.RESET}')

        start = self.editable_input(f'Start: ', last_end)
        end = self.editable_input(f'End: ', start)
        check = input(f'Is this correct? (y/n) [y]: ')
        if check == 'n' or check == 'N':
            start, end, _ = self.get_start_end(nb_demo, last_end)

        print(f'{Fore.YELLOW}[Demo {nb_demo}]{Fore.RESET} Start: {start} | End: {end}')

        start_timestamp = datetime.fromisoformat(start)
        start_timestamp = start_timestamp.replace(tzinfo=timezone.utc)
        start_timestamp = start_timestamp.isoformat()
        end_timestamp = datetime.fromisoformat(end)
        end_timestamp = end_timestamp.replace(tzinfo=timezone.utc)
        end_timestamp = end_timestamp.isoformat()

        return start_timestamp, end_timestamp, end
    
    def get_fps(self, video_path):
        count_f = 1
        last_s = None
        fpss = []
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            for side_data in frame.side_data:
                if side_data.type == 'S12M_TIMECODE':
                    raw_data = bytes(side_data)
                    h = self.bcd_to_dec(raw_data[4] & 0x7F)
                    m = self.bcd_to_dec(raw_data[5] & 0x7F)
                    s = self.bcd_to_dec(raw_data[6] & 0x7F)
                    f = self.bcd_to_dec(raw_data[7])
                    # print(f'{Fore.CYAN} [DEBUG]{Fore.RESET} Frame {i} | h: {h} | m: {m} | s: {s} | f: {f} | fpss: {fpss} | count_f: {count_f}')
            if s != last_s:
                fpss.append(count_f)
                last_s = s
                count_f = 1
            else:
                count_f += 1
            print(f'{Fore.RED} [fps extractor info]{Fore.RESET} frame {i}/400', end='\r')
            if i > 400:
                break
        print(f'{Fore.RED} [fps extractor info]{Fore.RESET} frame 400/400')
        return np.mean(fpss[2:])

    def create_video(self):
        container = av.open(self.video_path)
        output_video_name = self.video_path.split('\\')[-1].split('.')[0] + '_timed.avi'
        video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 1000) # 1920x1080
        fontScale = 1
        color = (0, 0, 0)
        thickness = 2

        i = 0
        for frame in container.decode(video=0):
            try:
                h, m, s, ms = self.extract_timestamps(frame)
                timestamp = self.to_iso8601(h, m, s, ms)
                frame_array = frame.to_ndarray(format='rgb24')
                image = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                image = cv2.putText(image, timestamp, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
                video.write(image)

                i += 1
                print(f'{Fore.CYAN} [video info]{Fore.RESET} Extracted frame {i}', end='\r')
            except OSError:
                print(f'{Fore.RED} [video warning]{Fore.RESET} Frame {i} could not be extracted')

        cv2.destroyAllWindows()
        video.release()
        print(f'{Fore.CYAN} [video info]{Fore.RESET} Video created and saved as {output_video_name}')

    def extract_demos(self):
        # Set the demo number (1 if no demos found)
        items = os.listdir(self.demos_path)
        nb_demo = [int(item.split('_')[-1]) for item in items if 'demo' in item]
        nb_demo = max(nb_demo)+1 if nb_demo else 1
        if nb_demo == 1:
            last_end = '2024-08-21T15:45:29'
            print(f'{Fore.GREEN} [demo info]{Fore.RESET} No demos found in {self.demos_path}, starting from demo 1')
        else:
            print(f'{Fore.GREEN} [demo info]{Fore.RESET} Found demos in {self.demos_path}, starting from demo {nb_demo}')
            dataframes = pd.read_csv(self.demos_path + f'/demo_{nb_demo-1}/index.csv')
            last_end = dataframes['timestamp'].iloc[-1]

        print(f'{Fore.GREEN} [demo info]{Fore.RESET} Please open the video {self.video_path} and write the start and end timestamps of each demonstration you want to keep')
        print(f'{Fore.LIGHTRED_EX} [demo warning]{Fore.RESET} The demos have to be in chronological order')
        
        # Create a dataframe to store the timestamps and frame names
        dataframe = pd.DataFrame(columns=['timestamp', 'frame'])
        container = av.open(self.video_path)

        start_timestamp, end_timestamp, last_end = self.get_start_end(nb_demo, last_end)
        frame_count = 0
        is_initial_pose = True
        for frame in container.decode(video=0):
            folder_path = self.demos_path + f'/demo_{nb_demo}'
            os.makedirs(folder_path, exist_ok=True)

            h, m, s, ms = self.extract_timestamps(frame)
            timestamp = self.to_iso8601(h, m, s, ms)

            if is_initial_pose:
                # Save the timestamp to a text file
                print(f'{Fore.GREEN} [demo info]{Fore.RESET} Saving initial pose timestamp to {self.demos_path}/initial_pose.txt')
                with open(self.demos_path + '/initial_pose.txt', 'w') as f:
                    f.write(timestamp)
                is_initial_pose = False

            if timestamp >= start_timestamp and timestamp <= end_timestamp:
                frame_array = frame.to_ndarray(format='rgb24')
                file_path = folder_path + f'/frame_{frame_count}.npy'
                np.save(file_path, frame_array)
                new_row = pd.DataFrame({'timestamp': [timestamp], 'frame': [f'frame_{frame_count}.npy']})
                dataframe = pd.concat([dataframe, new_row], ignore_index=True)
                print(f'{Fore.GREEN}[frame info]{Fore.RESET} Extracting frame {frame_count}', end='\r')

            elif timestamp > end_timestamp:
                dataframe.to_csv(folder_path + '/index.csv', index=False)
                dataframe = pd.DataFrame(columns=['timestamp', 'frame'])
                nb_demo += 1
                start_timestamp, end_timestamp, last_end = self.get_start_end(nb_demo, last_end)
                last_end = end_timestamp

            else:
                print(f'{Fore.GREEN}[frame info]{Fore.RESET} Passing frame {frame_count}...', end='\r')

            frame_count += 1
        
    def add_logs(self):
        """
        Gather the index.csv of each demo into one pandas DataFrame and add the logs from the database.
        """
        print(f'{Fore.BLUE} [logs info]{Fore.RESET} Adding logs to the demos... it is going to be LEGEN... Wait for it...')
        index_dataframe = pd.DataFrame(columns=['timestamp', 'frame'])
        
        demos_path = [f'{self.demos_path}/{demo}' for demo in os.listdir(self.demos_path) if 'demo' in demo]
        demos_path.sort(key=lambda x: int(x.split('_')[-1]))
        list_storing_length_demo = []

        for demo in demos_path:
            index_path = f'{demo}/index.csv'
            demo_dataframe = pd.read_csv(index_path)
            index_dataframe = pd.concat([index_dataframe, demo_dataframe], ignore_index=True)
            list_storing_length_demo.append(len(demo_dataframe))
        
        print(f'{Fore.BLUE} [logs info]{Fore.RESET} Found {len(demos_path)} demos in {self.demos_path}')

        # Load logs from database
        columns = ['unix_timestamp', 
                'robot_left_j1', 'robot_left_j2', 'robot_left_j3', 
                'robot_left_j4', 'robot_left_j5', 'robot_left_j6', 
                'robot_right_j1', 'robot_right_j2', 'robot_right_j3', 
                'robot_right_j4', 'robot_right_j5', 'robot_right_j6',
                'camera_north', 'camera_east']
        columns_str = ', '.join(columns)
        conn = sqlite3.connect(self.logs_path + '/device_frames.db3')
        query = f"SELECT {columns_str} FROM device_frame"
        df_logs_device_frame = pd.read_sql_query(query, conn)
        conn.close()

        index_dataframe['logs'] = None
        # Convert timestamps to datetime objects
        index_dataframe['timestamp'] = pd.to_datetime(index_dataframe['timestamp'], format='ISO8601')
        df_logs_device_frame['unix_timestamp'] = pd.to_datetime(df_logs_device_frame['unix_timestamp'], format='ISO8601')

        def find_closest_log(timestamp):
            idx = np.searchsorted(df_logs_device_frame['unix_timestamp'], timestamp, side="left")
            if idx > 0 and (idx == len(df_logs_device_frame) or 
                            abs(timestamp - df_logs_device_frame['unix_timestamp'].iloc[idx-1]) < 
                            abs(timestamp - df_logs_device_frame['unix_timestamp'].iloc[idx])):
                result = idx - 1
            
            else:
                result =  idx

            # Check if the difference in timestamps is less than 0.0333 seconds
            if abs(timestamp - df_logs_device_frame['unix_timestamp'].iloc[result]) > timedelta(seconds=0.0333):
                print(f'{Fore.RED} [logs warning]{Fore.RESET} Frame timestamp {timestamp} is more than 0.0333 seconds away from the closest log timestamp {df_logs_device_frame["unix_timestamp"].iloc[idx]}')
            
            return result
        
        closest_indices = index_dataframe['timestamp'].apply(find_closest_log)
        index_dataframe['logs'] = closest_indices.apply(lambda idx: df_logs_device_frame.iloc[idx][columns[1:]].to_dict())

        # Save each part of Dataframe in the correct demo folder
        for i, demo in enumerate(demos_path):
            demo_dataframe = index_dataframe.iloc[:list_storing_length_demo[i]]
            demo_dataframe.to_csv(f'{demo}/index.csv', index=False)
            index_dataframe = index_dataframe.iloc[list_storing_length_demo[i]:]
            print(f'{Fore.BLUE} [logs info]{Fore.RESET} Logs added to {demo}/index.csv')

        print(f'{Fore.BLUE} [logs info]{Fore.RESET} DARY, LEGENDARY!, Logs added to the demos.')

        # Get the logs of initial pose
        with open(self.demos_path + '/initial_pose.txt', 'r') as f:
            initial_pose_timestamp = f.read()
        initial_pose_timestamp = pd.to_datetime(initial_pose_timestamp, format='ISO8601')
        initial_pose_log = df_logs_device_frame.iloc[find_closest_log(initial_pose_timestamp)][columns[1:]].to_dict()
        print(f'{Fore.BLUE} [logs info]{Fore.RESET} Saved initial pose logs: {initial_pose_log}')
        # Save the pose to the text file
        with open(self.demos_path + '/initial_pose.txt', 'w') as f:
            f.write(str(initial_pose_log))

    def watch_demo(self, demo_number):
        """
        Take a demo folder and create a video from its frames in there, and download it.
        """
        demo_path = f'{self.demos_path}/demo_{demo_number}'
        frames = [f for f in os.listdir(demo_path) if f.endswith('.npy')]
        frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        video_name = f'{self.demos_path}/demo_{demo_number}.avi'
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
        for frame in frames:
            frame_array = np.load(f'{demo_path}/{frame}')
            image = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video.write(image)
            print(f'{Fore.CYAN} [video info]{Fore.RESET} Extracting frame {frame}/{len(frames)}', end='\r')
        video.release()


if __name__ == '__main__':
    video_path = '/data/gb/needle_pick_handoff_dataset_train/2024-08-27-15-09-29.h264'
    Extract_Demos(video_path, verbose=True)