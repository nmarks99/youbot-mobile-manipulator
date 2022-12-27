import numpy as np
import modern_robotics as mr 

def FeedbackControl(X,Xd,Xd_next, Kp, Ki, dt, Ierr):
    '''

    Args:
    -----
        X: Current actual end-effector configuration (Tse)
        Xd: Current end-effector reference configuration
        Xd_next: End-effector reference configuration at the next timestep
        Kp: Proportional gain matrix
        Ki: Integral gain matrix
        dt: Timestep between reference trajectory configurations
        Ierr: Integral error at last iteration
    
    Returns:
    --------
        V: Commanded end-effector twist V expressed in the end-effector frame

    '''
    
    # Compute reference twist from reference trajectory
    Xd_inv = mr.TransInv(Xd) 
    Vd_mat = 1/dt * mr.MatrixLog6(np.dot(Xd_inv,Xd_next))
    Vd = mr.se3ToVec(Vd_mat)

    # Compute error twist 
    Xinv = mr.TransInv(X) 
    xinv_dot_xd = np.dot(Xinv,Xd)
    Xerr_mat = mr.MatrixLog6(xinv_dot_xd)
    Xerr = mr.se3ToVec(Xerr_mat)

    # Compute adjoint to change representation of reference twist
    adj = mr.Adjoint(xinv_dot_xd)
    adj_Vd = np.dot(adj,Vd)

    # Compute commanded twist with Proportional-Integral controller
    Ierr += Xerr*dt
    V = adj_Vd + Kp @ Xerr + Ki @ Ierr

    return V, Xerr, Ierr
