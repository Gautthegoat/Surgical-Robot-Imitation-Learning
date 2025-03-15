import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

class Transform:
    def __init__(self, config):
        self.config = config

    def transform_frame(self, height, width, frame):
        """
        Resize, normalize and transform an array to tensor.
        """
        transform_first = A.Compose([
            A.Resize(height=height, width=width),  # Resize to standard ImageNet size
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=self.config['data']['random_BC']),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=self.config['data']['random_HSV']),
            A.GaussianBlur(blur_limit=(3, 7), p=self.config['data']['random_blur']),
            A.GaussNoise(var_limit=(10.0, 50.0), p=self.config['data']['random_noise'])
        ])

        if self.config['data']['frame_norm'].lower() == 'imagenet':
            transforme_norm = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225],   # ImageNet std
                    max_pixel_value=255.0
                ),  
                ToTensorV2()
            ])
        elif self.config['data']['frame_norm'].lower() == 'ZeroOne':
            transforme_norm = A.Compose([
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],  # Zero mean
                    std=[1.0, 1.0, 1.0],   # One std
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            raise ValueError(f"Frame normalization {self.config['data']['frame_norm']} not found.")
            
        transform = A.Compose([
            transform_first,
            transforme_norm
        ])

        transformed = transform(image=frame)

        return transformed['image']  # This is now a torch.Tensor of shape (3, 224, 224)

    def transform_actions(self, log):
        """
        Transform a log array to a tensor.
        """
        log_tensor = torch.from_numpy(log).float()

        return log_tensor



#############################################################################################################@
# MIRA STUFF NIXON DID THIS NEVER TRIED IT SO I DONT KNOW IF IT WORKS, good luck. 
#############################################################################################################@

class MiraTransform:

    # This class is a wrapper around the MIRA robot class to provide some useful functions for transforming between angles and XYZ coordinates.
    # Uncomment the following lines to use this class.

    # try:
    #     from vic.robots.mira import Mira
    #     import numpy as np
    # except ImportError:
    #     raise ImportError("Please install the MIRA package to use this class.")
        
    # mira = Mira()

    def get_matrices_from_angles(angles):
        '''Get the transformation matrix of the robot from the angles.'''
        left_angles = angles[:6]
        right_angles = angles[6:]
        # left_matrix = mira.left_arm._dh_robot.fkine(np.radians(left_angles))
        # right_matrix = mira.right_arm._dh_robot.fkine(np.radians(right_angles))
        left_matrix = mira.left_arm._dh_robot.fkine_all(np.radians(left_angles))[-1]
        right_matrix = mira.right_arm._dh_robot.fkine_all(np.radians(right_angles))[-1]
        return left_matrix, right_matrix

    def get_xyz_from_angles(angles):
        '''Get the XYZ coordinates of the robot from the angles.'''
        left_matrix, right_matrix = get_matrices_from_angles(angles)
        return np.concatenate((left_matrix.t, right_matrix.t))

    def get_position_from_angles(angles):
        '''Get the transformation matrix of the robot from the angles.'''
        left_matrix, right_matrix = get_matrices_from_angles(angles)

        left_xyz = left_matrix.t
        right_xyz = right_matrix.t

        left_rot = left_matrix.R
        right_rot = right_matrix.R

        # flatten the first two columns of the rotation matrix
        left_orientation = left_rot[:,:2].flatten()
        right_orientation = right_rot[:,:2].flatten()

        return np.concatenate((left_xyz, left_orientation, right_xyz, right_orientation))

    def get_delta_position_from_angles(angles1, angles2, reference_frame):
        '''Get the delta transformation matrix of the robot from the angles.'''
        if reference_frame == 'baselink':
            left_matrix1, right_matrix1 = get_matrices_from_angles(angles1)
            left_matrix2, right_matrix2 = get_matrices_from_angles(angles2)
            # TODO: i believe this is correct, but it needs to be tested
            left_diff = left_matrix1.inv() @ left_matrix2
            right_diff = right_matrix1.inv() @ right_matrix2
        elif reference_frame == 'tool':
            left_matrix1, right_matrix1 = get_matrices_from_angles(angles1)
            left_matrix2, right_matrix2 = get_matrices_from_angles(angles2)
            # TODO: i believe this is correct, but it needs to be tested
            left_diff = left_matrix2 @ left_matrix1.inv()
            right_diff = right_matrix2 @ right_matrix1.inv()

        left_xyz = left_diff.t
        right_xyz = right_diff.t

        left_rot = left_diff.R
        right_rot = right_diff.R

        # flatten the first two columns of the rotation matrix
        left_orientation = left_rot.as_matrix()[:,:2].flatten()
        right_orientation = right_rot.as_matrix()[:,:2].flatten()

        return np.concatenate((left_xyz, left_orientation, right_xyz, right_orientation))

    def get_angles_from_xyz(xyz):
        '''Get the angles of the robot from the XYZ coordinates.'''
        left_xyz = xyz[:3] + [0] # add a zero for the elbow angle
        right_xyz = xyz[3:] = [0] # add a zero for the elbow angle
        left_angles = mira.left_arm.ik_solve(left_xyz, polar=False).q
        right_angles = mira.right_arm.ik_solve(right_xyz, polar=False).q
        return np.concatenate((np.degrees(left_angles), np.degrees(right_angles)))