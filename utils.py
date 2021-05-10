import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def load_data(file_name, load_features = False):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
        load_features: a boolean variable to indicate whether to load features
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images,
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:

        t = data["time_stamps"] # time_stamps
        features = None

        # only load features for 03.npz
        # 10.npz already contains feature tracks
        if load_features:
            features = data["features"] # 4 x num_features : pixel coordinates of features

        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame

    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose,landmark, pose2, landmark2, path_name=["EKF_Motion_Visual","EKF VI SLAM"],show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name[0])
    ax.plot(pose2[0,3,:],pose2[1,3,:],'b',label=path_name[1])
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    plt.scatter(landmark[0],landmark[1], s = 0.5, alpha = 0.7)
    plt.scatter(landmark2[0],landmark2[1], s = 0.5, alpha = 0.7, c = 'purple')
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []

        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)

        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax

def init_car(number_of_scans, w ):
  data = {}
  data['mu'] = np.eye(4) #x,y,z,and angular velocity
  data['covar'] = w*np.eye(6) # x,y,z linear velocity, x,y,z angular velocity
  data['path'] = np.zeros([4,4, number_of_scans])
  return data

def init_landmarks(features, noise = 0.01):
  data = {}
  data['mu'] = np.empty((4, features))
  data['mu'].fill(np.nan)
  data['covar'] = np.zeros((3,3,features)) *.001
  for i in range(features):
    data['covar'][:, :, i] = noise * np.eye(3)
  return data

def create_skew_sym_matrix(x):
  x = np.squeeze(x)
  symm = np.array([[0,         -x[2],    x[1]],
                      [x[2],      0,        -x[0]],
                      [-x[1],     x[0],     0   ]])
  return symm

def create_skew_sym_matrix_big(x):
    t = x[3:, np.newaxis]
    p = x[:3, np.newaxis]
    symm = np.block([[create_skew_sym_matrix(t), -p],
                         [np.zeros((1, 4))]])
    return symm

def EKF_motion_model(car, linear_velocity, angular_velocity, timestamp_delta, noise_v1, noise_w):
  lin_v = linear_velocity.reshape(3,1)
  # covariance for movement noise
  noise_matrix = np.block([[noise_v1 * np.eye(3), np.zeros((3,3))],
                           [np.zeros((3, 3)), noise_w * np.eye(3)]])
  u_hat = np.vstack((np.hstack((create_skew_sym_matrix(angular_velocity),
                               lin_v)),
                               np.zeros((1, 4))))
  car['mu'] = car['mu'] @ expm(timestamp_delta * u_hat)
  u_hat_cov = np.block([[  create_skew_sym_matrix(angular_velocity),     create_skew_sym_matrix(linear_velocity)],
                         [  np.zeros((3, 3)), create_skew_sym_matrix(angular_velocity)]])
  car['covar'] = expm(-1*timestamp_delta * u_hat_cov) @ car['covar'] @ np.transpose(expm(-1*timestamp_delta * u_hat_cov)) + noise_matrix

def stereo_cam_calib_matrix(K, stereo_baseline, ):
  M = np.array([[K[0, 0],         0, K[0, 2],            0],
              [      0,   K[1, 1], K[1, 2],            0],
              [K[0, 0],         0, K[0, 2], -K[0, 0] * stereo_baseline],
              [      0,   K[1, 1], K[1, 2],            0]])
  return M

# feature to cam
def cam_T_feature(features, K, stereo_baseline):
  M = stereo_cam_calib_matrix(K, stereo_baseline)
  features = features.T
  # print('here is len features', len(features))
  U_l = features[0] #x left
  V_l = features[1] #y left
  U_r = features[2] #x right
  # V_r = features[3] #y right
  z = K[0][0] * stereo_baseline / (U_l - U_r)
  x = (U_l - M[0][2])*z / M[0][0]
  y = (V_l - M[1][2])*z / M[1][1]  #these are from hw2
  return x, y, z

#U_l = f_su * 1/z * x + C_u
# x = (U_l - C_u)*z / f_su
# x = (U_l - K_13)*z / (K_22)
#V_l = f_sv* 1/z * y + C_v
# y = (V_l - C_v)*z / f_sv
#U_l - U_r = 1/z * f_su_b
#z = f_su_b / (U_l - U_r)

def projection_func(q):
  return q/q[2]

def projection_func_deriv(q):
  d =  np.block([[1, 0, -q[0]/q[2], 0],
                         [0, 1, -q[1]/q[2], 0],
                         [0,0,0,0],
                         [0,0,-q[3]/q[2], 1]
                         ])
  return d / q[2]



def EKF_landmarks_update(feature, landmarks, car, K, b, imu_T_cam, trust_v ):
  cam_T_imu = np.linalg.inv(imu_T_cam)
  M = stereo_cam_calib_matrix(K, b)       #M is calib matrix
  world_T_imu = (car['mu'])   # inverse of car pose is imu to world
  world_T_cam = world_T_imu @ imu_T_cam
  p = np.eye(3, 4)
  for i in range(13289):
    feat = feature[:,i][:]
    # print('here is shape of feat', feat.shape)
    if np.all(feat == -1):
      continue
    # print('here is landmarks mu', landmarks['mu'])
    if np.all(np.isnan(landmarks['mu'][:, i])) : #when we havent seen a landmark
      # x,y,z = cam_T_feature(feat, K, b)
      disparity = (feat[0] - feat[2])
      z = (K[0, 0] * b) / disparity
      # coord = np.append(np.append(x,y),  1)
      # homog_coord = np.append(z * np.linalg.inv(K) @ coord, 1)
      cam = np.hstack((z * np.linalg.inv(K) @ np.hstack((feat[:2], 1)), 1)) #landmark to camera
      landmarks['mu'][:, i] = world_T_cam @ cam
      continue
    cam_T_world = cam_T_imu @ np.linalg.inv(car['mu']) #car['mu'] is imu_T_world
    land = cam_T_world @ landmarks['mu'][:, i]
    z_pred = M @ projection_func(land) #predicted observation slide 13_18
    #jacobian of z_pred slide 13_18 w/ resp to world_T_imu
    #is the negative 1 correct?
    # print('here is size of cam_t_world', cam_T_world.shape)
    # print('here is size of projection_func_deriv(land)', projection_func_deriv(land).shape)
    # print('here is size of M', M.shape)
    # H = -1*M @ projection_func_deriv(land) @ cam_T_world @ np.eye(4,3)
    H = M @ projection_func_deriv(land) @ cam_T_world @ p.T
    kalman_g = landmarks['covar'][:, :, i] @ H.T @ np.linalg.inv(H @ landmarks['covar'][:, :, i] @  H.T+ np.eye(4)*trust_v)
    landmarks['mu'][:,i] = landmarks['mu'][:,i] + p.T @ kalman_g @ (feat - z_pred) #slide 13_18
    landmarks['covar'][:, :, i] =  (np.eye(3) - kalman_g @ H) @ landmarks['covar'][:, :, i] # slide 13_18

def EKF_VI_prediction(car_vi, linear_velocity, angular_velocity, timestep_delta, noise_v, noise_w):
    W = np.block([[noise_v * np.eye(3), np.zeros((3,3))],
                  [    np.zeros((3, 3)), noise_w * np.eye(3)]])

    u_hat = np.vstack((np.hstack((create_skew_sym_matrix(angular_velocity), linear_velocity.reshape(3, 1))), np.zeros((1, 4))))
    u_curlyhat = np.block([[create_skew_sym_matrix(angular_velocity),     create_skew_sym_matrix(linear_velocity)],
                           [np.zeros((3, 3)), create_skew_sym_matrix(angular_velocity)]])
    #slide 14
    car_vi['mu'] = car_vi['mu'] @ expm(timestep_delta * u_hat)  # + noise
    car_vi['covar'] = expm(-timestep_delta * u_curlyhat) @ car_vi['covar'] @ np.transpose(expm(-timestep_delta * u_curlyhat)) + W


def EKF_VI_update(feature, landmarks, car, K, b, imu_T_cam, noise_vis ):
  # T_t^-1 is world to imu
  V = np.eye(4) * noise_vis  #V is 4x4 covar of visual noise, slide 5
  T_t_inv = np.linalg.inv(car['mu'])
  cam_T_imu = np.linalg.inv(imu_T_cam)
  M = stereo_cam_calib_matrix(K, b)       #M is calib matrix
  p = np.eye(3, 4)
  world_T_imu = (car['mu'])   # inverse of car pose is imu to world
  world_T_cam = world_T_imu @ imu_T_cam
  for i in range(feature.shape[1]):
    feat = feature[:,i]
    if np.all(feat == -1):
      continue
    if (np.all(np.isnan(landmarks['mu'][:, i]))): #when we havent seen a landmark
      # x,y,z = cam_T_feature(feat, K, b)
      # coord = np.append(np.append(x,y),  1)
      # homog_coord = np.append(z * np.linalg.inv(K), coord)
      # landmarks['mu'][:,i] = world_T_cam @ homog_coord
      # continue
      d = (feat[0] - feat[2])
      Z_0 = (K[0, 0] * b) / d
      camera_frame_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((feat[:2], 1)), 1))
      landmarks['mu'][:, i] = world_T_cam @ camera_frame_coords
      continue
    cam_T_world = cam_T_imu @ car['mu']
    land = cam_T_world @ landmarks['mu'][:, i]
    z_pred = M @ projection_func(land) #predicted observation slide 13_18
    #jacobian of z_pred slide 13_18 w/ resp to world_T_imu
    H = -1*M @ projection_func_deriv(land) @ cam_T_imu @ np.block([[np.eye(3), -create_skew_sym_matrix(feat[:3])], [np.zeros((1, 6))]])
    kalman_g = car['covar'] @ H.T @ np.linalg.inv(H @ car['covar'] @ H.T + V)
    car['mu'] = car['mu'] @ expm(create_skew_sym_matrix_big(kalman_g @ (feat - z_pred)))
    car['covar'] = (np.eye(6) - kalman_g @ H) @ car['covar'] #slide 18
    # pi = projection_func(feat)
    # pi_d = projection_func_deriv(feat)







