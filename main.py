import numpy as np
from utils import load_data, init_car, init_landmarks,EKF_motion_model, EKF_landmarks_update,visualize_trajectory_2d,EKF_VI_prediction, EKF_VI_update


if __name__ == '__main__':

	# only choose ONE of the following data

	# data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part
    filename = "./data/10.npz"
    t,features,linear_velocity,angular_velocity, K, b, imu_T_cam = load_data(filename, load_features = True)
    scans = len(t[0])
    trust_v = 0.000000001
    trust_w = 0.00000001
    #t = timestamp in UNIX standard
    #b = stereo baseline
    #K = camera calibration matrix
    #imu_T_cam rotational transformation from left camera to IMU

	# data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra credit
	#filename = "./data/03.npz"
	#t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    car = init_car(scans, 0.01)
    landmarks = init_landmarks(features.shape[1])
    car_vi = init_car(scans, 0.01)
    landmarks_vi = init_landmarks(features.shape[1])
	# (a) IMU Localization via EKF Prediction
    for i in range(1, scans):
      timestep_delta = t[0][i] - t[0][i-1]
      EKF_motion_model(car, linear_velocity[:,i], angular_velocity[:,i], timestep_delta, trust_v, trust_w)
      # print('car mu is', car['mu'][:2,3])
      car['path'][:,:,i] = car['mu']
	# (b) Feature detection and matching

	# (c) Landmark Mapping via EKF Update
    # slide 28
      EKF_landmarks_update(features[:, :, i], landmarks, car, K, b, imu_T_cam, 1000 )
	# (d) Visual-Inertial SLAM
      EKF_VI_prediction(car_vi, linear_velocity[:,i], angular_velocity[:,i], timestep_delta, trust_v, trust_w)
      EKF_VI_update(features[:, :, i], landmarks_vi, car_vi, K, b, imu_T_cam, 2550000000 ) #try 10k next
      # print('car_vi mu is', car_vi['mu'][:2,3])
      car_vi['path'][:,:,i] = car_vi['mu']
	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
      if ((i - 1) % 100 == 0 or i == t.shape[1] - 1):
        visualize_trajectory_2d(car['path'], landmarks['mu'], car_vi['path'], landmarks_vi['mu'])
        # visualize_trajectory_2d(, 'EKF VI SLAM')
