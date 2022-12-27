#!/usr/bin/env python3
import numpy as np
import sys
from numpy import sin, cos
from scipy.spatial.transform import Rotation
import modern_robotics as mr 

from feedback import FeedbackControl
from traj import TrajectoryGenerator 
from step import NextState
from utils import Tse_from_config, compute_controls
from utils import plot_err


CUBE_WIDTH = 0.05

# Get command line arguments to decide which to demo
assert(len(sys.argv) == 2), "No argument given"
ARG = sys.argv[1]


if ARG == "best" or ARG == "overshoot":

    # Starting configuration of the cube
    # Rotate 90deg about y-axis so gripper is oriented correctly
    R = Rotation.from_euler("y",135,degrees=True).as_matrix()
    cube_start = mr.RpToTrans(R, p=np.array([1.02,0.0,0.0+CUBE_WIDTH/2]))

    # Goal configuration of the cube
    # Rotate an additional -90 about z so gripper is correct
    R = Rotation.from_euler("yz",(135,-90),degrees=True).as_matrix()
    cube_goal = mr.RpToTrans(R, p=np.array([0.0,-1.0,0.0+CUBE_WIDTH/2]))

    # Initial robot configuration
    robot_config = np.array([
        0,0,0,
        0,0,0,0,0,
        0,0,0,0
    ])


if ARG == "best":
    # Controller gains
    Kp = np.eye(6)*2.0
    Ki = np.eye(6)*0.03
    Ierr = np.zeros(6)

    traj_filename = "../results/best/trajectory.csv"
    err_plots_filename = "../results/best/error.png"
    err_data_filename = "../results/best/error.csv"
    
    # Initial robot configuration
    robot_config = np.array([
        0,0,0,
        0,0,0,0,0,
        0,0,0,0
    ])
    max_speed = 15.0


elif ARG == "overshoot":
    # Controller gains
    Kp = np.eye(6)*0.8
    Ki = np.eye(6)*0.5
    Ierr = np.zeros(6)

    traj_filename = "../results/overshoot/trajectory.csv"
    err_plots_filename = "../results/overshoot/error.png"
    err_data_filename = "../results/overshoot/error.csv"

    # Initial robot configuration
    robot_config = np.array([
        0,0,0,
        0,0,-np.pi/4,-np.pi/6,0,
        0,0,0,0
    ])
    max_speed = 15.0


elif ARG == "new":
    # Starting configuration of the cube
    # Rotate 90deg about y-axis so gripper is oriented correctly
    R = Rotation.from_euler("y",135,degrees=True).as_matrix()
    cube_start = mr.RpToTrans(R, p=np.array([0.75,-0.6,0.0+CUBE_WIDTH/2]))

    # Goal configuration of the cube
    # Rotate an additional -90 about z so gripper is correct
    R = Rotation.from_euler("yz",(135,-90),degrees=True).as_matrix()
    cube_goal = mr.RpToTrans(R, p=np.array([0.0,-1.0,0.0+CUBE_WIDTH/2]))
    
    # Controller gains
    Kp = np.eye(6)*2.0
    Ki = np.eye(6)*0.03
    Ierr = np.zeros(6)

    traj_filename = "../results/newTask/trajectory.csv"
    err_plots_filename = "../results/newTask/error.png"
    err_data_filename = "../results/newTask/error.csv"
    
    # Initial robot configuration
    robot_config = np.array([
        0,0,0,
        0,0,0,0,0,
        0,0,0,0
    ])
    max_speed = 15.0



# Initial point in reference trajectory
X = Tse_from_config(robot_config)

# Generate the reference trajectory and save it to a CSV file
ref_traj_SE3, ref_traj_vec = TrajectoryGenerator(X, cube_start, cube_goal)

dt = 0.01 # timestep

print("Running main loop.")
robot_state_history = []
Xerr_history = []
for k in range(len(ref_traj_SE3)-1):

    # Gripper state is last element of reference trajectory vector
    gripper_state = ref_traj_vec[k][-1]

    # Get Xd, which is the reference configuration of the EE at kth step    
    Xd = ref_traj_SE3[k]

    # Get Xd_next, which is reference configuration of the EE at k+1 step
    Xd_next = ref_traj_SE3[k+1]

    # Run Feedforward + PI controller to get commanded twist and current error
    V, Xerr, Ierr = FeedbackControl(X,Xd,Xd_next,Kp,Ki,dt,Ierr)
    
    # Get control velocities from the commanded twist and current arm joint angles
    controls = compute_controls(V,robot_config[3:8]) 

    # Get the robot configuration at the next step as a result of the control velocities
    robot_config = NextState(robot_config,controls,timestep=dt,max_angular_speed=max_speed)

    # Compute Tse (X) for the new robot configuration
    X = Tse_from_config(robot_config)

    # append the gripper state then save it to robot_state_history arr
    robot_state_history.append(np.append(robot_config,gripper_state))
    Xerr_history.append(Xerr)

# Save trajectory to CSV file
np.savetxt(traj_filename,np.array(robot_state_history),delimiter=",")  

# Save error history to CSV file
np.savetxt(err_data_filename,np.array(Xerr_history),delimiter=",")  

# Generate error plots
plot_err(Xerr_history,dt=dt,filename=err_plots_filename)
print("Done.")


