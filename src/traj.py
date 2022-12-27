
import modern_robotics as mr
import numpy as np

SIM_TSTEP = 0.01
V_MAX = 0.75
GRIPPER_TIME_MULT = 63


def _get_vec13(T,gripper_state="OPEN"):
    '''
    Generate a length 13 array of the following form:
    [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state]

    given an SE(3) transformation matrix T of the form 
    [r11, r12, r13, px],
    [r21, r22, r23, py],
    [r31, r32, r33, pz],
    [  0,   0,   0, 1]

    and append a 1.0 or 0.0 to the end for the gripper state with 0=open, 1=close
    '''

    R,p = mr.TransToRp(T)
    R = R.flatten()
    if gripper_state == "OPEN":
        gripper_state = 0.0
    elif gripper_state == "CLOSE":
        gripper_state = 1.0
    vec13 = np.concatenate([R,p,np.array([gripper_state])])
    return vec13 


def _get_time_multiplier(T1, T2):
    '''
    Gets the linear distance travelled through the transformation between
    T1 and T2 and computes the time multiplier to determine the total time 
    for a single trajectory, taking into account max velocity
    '''

    # this really doesn't make a whole lot of sense but it yields some 
    # believable speeds so I'm gonna leave it
    _,p12 = mr.TransToRp(np.dot(T1,T2))
    d = np.linalg.norm(p12)
    n = np.floor(d/(V_MAX*SIM_TSTEP))
    return n


def TrajectoryGenerator(Tse_initial, T_cube_start, T_cube_goal,standoff=(0.0,0.0,0.1), method=5):
    '''
    Computes a trajectory for the gripper for picking up the cube
    and placing it at the goal location.
    
    Args:
    -----

        Tse_initial : Starting configuration of the end-effector in the s frame in SE(3)
        
        T_cube_start: Starting configuration of the cube in the s frame in SE(3)
        
        T_cube_goal: Goal configuration of the cube in the s frame in SE(3)
        
        standoff (optional): Standoff position of the end-effector in the
        cube frame given as a 3-vector = (x,y,z). Will default to (0,0,0.1m)
        if unspecified

    Returns:
    --------
        traj_SE3: an SE(3) representation of the end-effector at each step along
        the trajectory
        traj_csv: a 13-vector corresponding to the R and p elements of the SE(3)
        matrix at each step as well as a 13th value corresponding to the state
        of the gripper, 1=closed, 0=open

    '''
    
    print("Generating reference trajectory.")

    # Create SE(3) matrices for standoff configurations at start and goal
    R, p = mr.TransToRp(T_cube_start)
    p = p + np.array(standoff)
    T_standoff_start = mr.RpToTrans(R, p)

    R, p = mr.TransToRp(T_cube_goal)
    p = p + np.array(standoff)
    T_standoff_goal = mr.RpToTrans(R, p)
        

    GRIPPER_STATE = "OPEN" # keeps track of the current gripper state
    
    # lists to store the entire trajectory 
    traj_csv = []
    traj_SE3 = []

    # list to store start,end configuration pairs for each of the 8 trajectory segments 
    waypoints = [] 
    

    # 1. start -> standoff 
    waypoints.append([Tse_initial,T_standoff_start])

    # 2. Move to pickup (start) location
    waypoints.append([T_standoff_start,T_cube_start])

    # 3. close the gripper
    waypoints.append([T_cube_start,"CLOSE"])      

    # 4. move back to standoff
    waypoints.append([T_cube_start,T_standoff_start])

    # 5. move to standoff above goal location
    waypoints.append([T_standoff_start,T_standoff_goal])
    
    # 6. Move to dropoff (goal) location
    waypoints.append([T_standoff_goal,T_cube_goal])

    # 7. open the gripper
    waypoints.append([T_cube_goal,"OPEN"])

    # 8. move back to standoff
    waypoints.append([T_cube_goal,T_standoff_goal])


    # Generate trajectories for each start,end pair in waypoints array
    for step in waypoints:
        if isinstance(step[1], str):
            GRIPPER_STATE = step[1]
            for i in range(GRIPPER_TIME_MULT):
                traj_SE3.append(step[0])
                vec13 = _get_vec13(step[0],gripper_state=step[1])
                traj_csv.append(vec13)

        else:
            Tf = SIM_TSTEP*_get_time_multiplier(step[0],step[1])
            res = mr.CartesianTrajectory(
                Xstart = step[0],
                Xend = step[1],
                Tf = Tf,
                N = Tf/SIM_TSTEP,
                method = method
            )
            for T in res:
                traj_SE3.append(T)
                vec13 = _get_vec13(T,gripper_state=GRIPPER_STATE)
                traj_csv.append(vec13)
    
    return np.array(traj_SE3), np.array(traj_csv)







