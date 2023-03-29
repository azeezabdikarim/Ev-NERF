import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# create 3x3 roation matrix from quaternion
def quat2rot(q):
    qx, qy, qz, qw = q[4], q[5], q[6], q[7]
    norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    R = np.array([[1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
                    [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
                    [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]])
    return R

#create 4x4 orientation matrix from quaternion
def quat2mat(q):
    px, py, pz = q[1], q[2], q[3]
    R = quat2rot(q)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.array([px, py, pz])
    return T

def getFocalLength(calib_file):
    # Read calibration matrix from file
    with open(calib_file, 'r') as f:
        calib = [float(x) for x in f.readline().split()]
    # Extract focal length
    fx, fy = calib[:2]
    focal_length = np.mean([fx, fy])  # Use mean of fx and fy as focal length
    return focal_length

def cameraKmatrix(calib_file):
    with open(calib_file, 'r') as f:
        params = f.readline().split()
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, params)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K

# Extract focal length
# fx, fy = calib[:2]
# focal_length = np.mean([fx, fy])  # Use mean of fx and fy as focal length
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_uzh_data():
    events = pd.read_csv('box_data/hdr_boxes/events.txt', sep=' ', header=None, names = ['t', 'x', 'y', 'p'])
    width = events.x.max() + 1
    height = events.y.max() + 1

    imgs = pd.read_csv('box_data/hdr_boxes/images.txt', sep=' ', header=None, names = ['t', 'file_name'])
    gt = pd.read_csv('box_data/hdr_boxes/groundtruth.txt', sep=' ', header=None, names = ['t', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

    gt['orientation_matrix'] = gt.apply(quat2mat, axis=1)

    #build dataset
    seed = 42
    np.random.seed(seed)
    interval = np.random.uniform(2, 4)
    print(f"The size of interval is {interval} seconds")
    packets = []

    start_time_interval = 0
    end_time_interval = interval
    count = 1

    while start_time_interval < imgs.t.max():
        interval_imgs = imgs[(imgs.t >= start_time_interval) & (imgs.t < end_time_interval)]
        interval_imgs['t2'] = interval_imgs['t'].shift(-1)
        interval_imgs['file_name_2'] = interval_imgs['file_name'].shift(-1)

        closest_pose = []
        for t in interval_imgs['t']:
            closest_index = np.abs(gt.t - t).argmin()
            closest_pose.append(gt.iloc[closest_index]['orientation_matrix'])
        interval_imgs['t_pose'] = closest_pose
        interval_imgs['t2_pose'] = interval_imgs['t_pose'].shift(-1)
        #remove the last line of the data frame since it contains NaNs
        interval_imgs = interval_imgs[:-1]
        packets.append(interval_imgs)

        print(f"Packet {count} is composed of {len(interval_imgs)} event windows")
        start_time_interval = end_time_interval
        end_time_interval += interval
        count += 1
    
    dataset_windows = []
    pose_pairs = []
    events_list = []
    image_pairs = []

    for i in range(len(packets)):
    # for i in range(1):
    # for i in [1,5,8]:
        p = packets[i]
        windows = []
        for wi in range(len(p)):
            #get one window description
            wd = p.iloc[wi]
            data = {}
            data['t1'] = wd['t']
            data['t2'] = wd['t2']
            data['img1'] = wd['file_name']
            data['img2'] = wd['file_name_2']
            data['p1'] = wd['t_pose']
            data['p2'] = wd['t2_pose']
            data['events'] = events.loc[(events['t'] >= wd['t']) & (events['t'] < wd['t2'])].values
            events_list.append(data['events'])
            pose_pairs.append([data['p1'], data['p2']])
            windows.append(data)
            image_pairs.append([data['img1'], data['img2']])
        dataset_windows.append(windows)

    indexes = list(range(len(pose_pairs)))
    i_train, i_val, i_test = np.split(indexes, [int(.7*len(indexes)), int(.85*len(indexes))])
    i_split = [i_train, i_val, i_test]
    focal_length = getFocalLength('box_data/hdr_boxes/calib.txt')
    hwf = [height, width, focal_length]
    K = cameraKmatrix('box_data/hdr_boxes/calib.txt')
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)


    return events_list, pose_pairs, render_poses, image_pairs, hwf, i_split, K

def visualize_events(events, width, height, time_window=0.1):
    """
    Visualize events from an event camera.
    
    Args:
        events (numpy array): A numpy array of events, shape (N, 4), where N is the number of events and
                              each event is represented as (timestamp, x, y, polarity).
        width (int): Width of the event camera sensor (number of pixels).
        height (int): Height of the event camera sensor (number of pixels).
        time_window (float): Time window to accumulate events for visualization, in seconds.
    """
    start_time = events[0, 0]
    end_time = events[-1, 0]

    current_time = start_time
    event_index = 0

    while current_time < end_time:
        event_image = np.zeros((height, width), dtype=np.int8)

        while event_index < len(events) and events[event_index, 0] < current_time + time_window:
            _, x, y, polarity = events[event_index]
            event_image[int(y), int(x)] = 1 if polarity > 0 else -1
            event_index += 1

        plt.imshow(event_image, cmap='gray', vmin=-1, vmax=1)
        plt.title(f'Events between {start_time:.4f}s and {end_time:.4f}s')
        plt.pause(0.1)

        current_time += time_window


