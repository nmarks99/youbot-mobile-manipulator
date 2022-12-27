import numpy as np
import modern_robotics as mr
from numpy import sin, cos
from matplotlib import pyplot as plt


# Define some constants associated with the youBot
M = np.array([
    [1.0, 0.0, 0.0, 0.033],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.6546],
    [0.0, 0.0, 0.0, 1.0]
])

Blist = np.array([
    [0.0,0.0,1.0,0.0,0.033,0.0],
    [0.0,-1.0,0.0,-0.5076,0.0,0.0],
    [0.0,-1.0,0.0,-0.3526,0.0,0.0],
    [0.0,-1.0,0.0,-0.2176,0.0,0.0],
    [0.0,0.0,1.0,0.0,0.0,0.0]
]).T

Tb0 = np.array([
    [1.0, 0.0, 0.0, 0.1662],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0026],
    [0.0, 0.0, 0.0, 1.0]
])

r = 0.0475
l = 0.47/2
w =0.3/2
F = r/4*np.array([
    [-1/(l+w),1/(l+w),1/(l+w),-1/(l+w)],
    [1.0,1.0,1.0,1.0],
    [-1.0,1.0,-1.0,1.0]
])
F6 = np.row_stack([np.zeros(4),np.zeros(4),F,np.zeros(4)])


def plot_err(Xerr_history,dt,filename="err.png"):
    '''
    Generates plots of the error history with the specified timestep.
    The figure is saved to the current working directory and contains
    6 individual plots corresponding to the error for the x, y, z, roll,
    pitch, and yaw. 
    '''
    
    print("Generating error plot data.")

    # Fill in some lists for the 6 error values stored in Xerr_history
    x_err = []
    y_err = []
    z_err = []
    roll_err = []
    pitch_err = []
    yaw_err = []
    for err in Xerr_history:
        x_err.append(err[0])
        y_err.append(err[1])
        z_err.append(err[2])
        roll_err.append(err[3])
        pitch_err.append(err[4])
        yaw_err.append(err[5])

    # Create the time vector
    t = dt*np.linspace(0,len(x_err),len(x_err))
    
    # generate the 6 subplots
    plt.style.use("seaborn")
    fig, ax = plt.subplots(2,3,figsize=(16,11))
    ax[0,0].plot(t,x_err)
    ax[0,1].plot(t,y_err)
    ax[0,2].plot(t,z_err)
    ax[1,0].plot(t,roll_err)
    ax[1,1].plot(t,pitch_err)
    ax[1,2].plot(t,yaw_err)

    # Create titles and ylabels 
    titles = [
        ["x error", "y error", "z error"],
        ["Roll error","Pitch error", "Yaw error"]
    ]
    
    ylabels = [
        ["Error (m)", "Error (m)", "Error (m)"],
        ["Error (rad)","Error (rad)", "Error (rad)"]
    ]

    plt.subplots_adjust(hspace=0.25) # adjust spacing
    for i in range(2): # apply labels
        for j in range(3):
            ax[i,j].set(
                xlabel="Time(s)",
                title = titles[i][j],
                ylabel = ylabels[i][j]
            )

    plt.savefig(filename) # save the figure to ./filename
    plt.show()

    plt.plot(Xerr_history)
    plt.show()

def Tse_from_config(robot_config):
    '''
    Computes the configuration of the end-effector Tse given the 
    configuration of the robot, usually as a 12-vector. If wheel
    angles are ommited that is fine too since they are not used.
    '''

    if len(robot_config) == 12:
        robot_config = robot_config[0:8]
    elif len(robot_config) != 8:
        raise ValueError("Robot config must be of length 12 or 8")

    phi = robot_config[0]
    x = robot_config[1]
    y = robot_config[2]
    theta = robot_config[3:] # joint angles of the arm

    # Use FKinBody to get end effector config in 0 frame
    T0e = mr.FKinBody(M,Blist,theta)

    # Compute Tsb
    Tsb = np.array([
        [cos(phi), -sin(phi), 0.0, x],
        [sin(phi), cos(phi), 0.0, y],
        [0.0, 0.0, 1.0, 0.0963],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Transform Tsb to Tse and return
    Ts0 = np.dot(Tsb,Tb0)
    Tse = np.dot(Ts0,T0e)
    return Tse


def compute_controls(V,thetalist):
    '''
    Computes the control velocities for the wheels and joints given the 
    commanded twist from the feedback controller.
    Args:
    -----
        V: Commanded end-effector twist V expressed in the end-effector frame
        thetalist: list of current arm joint angles

    Returns:
    --------
        controls: 9-vector [u, thetadot] of wheel and joint speeds computed from 
        the commanded velocity output from the feedback controller
    ''' 


    T0e = mr.FKinBody(M,Blist,thetalist)
    Jarm = mr.JacobianBody(Blist,thetalist)
    Jbase = np.dot(
            mr.Adjoint(np.dot(mr.TransInv(T0e),mr.TransInv(Tb0))),
            F6
    )
    Je = np.column_stack([Jbase,Jarm])
    pinv_Je = np.linalg.pinv(Je)
    controls = np.dot(pinv_Je,V)
    return controls


def compute_chassis_twist(wheel_config, new_wheel_config):
    '''
    Computes the body twist of the chassis at next step

    Args:
    -----
        wheel_config: 4-vector of wheel angles at current step
        new_wheel_config: 4-vector of wheel angles at next step

    Returns:
    --------
        Vb3: 3-vector body twist (w,vx,vy). Can be expressed as a 6-vector
        by rewriting it as Vb6 = (0,0,w,vx,vy,0)
    '''

    delta_theta = new_wheel_config - wheel_config
    Vb3 = np.dot(F,delta_theta)
    
    return Vb3


def compute_odometry(chassis_config, Vb3):
    '''
    Computes the chassis configuration at the next step 
    
    Args:
    -----
        chassis_config: 3-vector representing the (phi,x,y) configuration of the chassis
        Vb3: 3-vector representing the body twist of the chassis

    Returns:
    --------
        new_chassis_config: Chassis configuration at the next timestep

    '''
    
    wbz = Vb3[0]
    vbx = Vb3[1]
    vby = Vb3[2]
    if wbz == 0.0:
        delta_qb = np.array([0, vbx, vby])
    else:
        delta_qb = np.array([
            [wbz],
            [(vbx*sin(wbz) + vby*(cos(wbz) - 1))/wbz],
            [(vby*sin(wbz) + vbx*(1 - cos(wbz)))/wbz]
        ])
        delta_qb = delta_qb.reshape(3)
    
    phi_k = chassis_config[0]
    R = np.array([
        [1, 0, 0],
        [0, cos(phi_k), -sin(phi_k)],
        [0, sin(phi_k), cos(phi_k)]
    ])
    delta_q = np.dot(R,delta_qb)
    new_chassis_config = chassis_config + delta_q

    return new_chassis_config
