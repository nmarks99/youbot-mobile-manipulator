import numpy as np
from utils import compute_chassis_twist, compute_odometry


def NextState(robot_config, speeds, timestep=0.01, max_angular_speed=None):
    '''

    Args:
    -----

        robot_config (ndarray): 12-vector representing the current configuration
        of the robot, including 3 variables for the chassis, 5 for the 
        arm, and 4 for the wheel angles
        speeds (ndarray): 9-vector containing 4 wheel speeds and 5 arm joint speeds
        timestep (float): A timestep 
        max_angular_speed (float): Maximum angular speed of the joints and wheels

    Returns:
    --------
        
        next_state (ndarray): 12-vector representing the configuration of the 
        robot at the next timestep

    '''
    # Extract chassis, arm, and wheel config from robot_config vector
    chassis_config = robot_config[0:3]
    arm_config = robot_config[3:8]
    wheel_config = robot_config[8:]
    wheel_speeds = speeds[0:4]
    arm_speeds = speeds[4:]
    
    # Limit wheel and joint velocities if desired 
    if max_angular_speed is not None: 
        wheel_speeds[wheel_speeds > max_angular_speed] = max_angular_speed
        wheel_speeds[wheel_speeds < -max_angular_speed] = -max_angular_speed
        arm_speeds[arm_speeds > max_angular_speed] = max_angular_speed
        arm_speeds[arm_speeds < -max_angular_speed] = -max_angular_speed
    
    # Compute new arm and wheel configurations with 1st order Euler step
    new_arm_config = arm_config + arm_speeds*timestep
    new_wheel_config = wheel_config + wheel_speeds*timestep

    # Compute the chassis twist at the next step 
    Vb3 = compute_chassis_twist(wheel_config, new_wheel_config)

    # Compute odometry to find new chassis configuration
    new_chassis_config = compute_odometry(chassis_config,Vb3)

    # Concatenate the chassis, arm, and wheel configurations into a new 12-vector
    new_robot_config = np.array([*new_chassis_config,*new_arm_config,*new_wheel_config])
    
    return new_robot_config


