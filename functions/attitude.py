
import numpy as np
DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi
def quaternion_multiply(q_left,q_right):
    w1,x1,y1,z1=q_left
    w2,x2,y2,z2=q_right
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ])
def propagate_quaternion(q,omega_rad_s,dt_s):
    rate=np.linalg.norm(omega_rad_s)
    if rate==0.0:
        return q.copy()
    half=0.5*rate*dt_s
    axis=omega_rad_s/rate
    dq=np.concatenate(([np.cos(half)],axis*np.sin(half)))
    q=quaternion_multiply(q,dq)
    return q/np.linalg.norm(q)
def quaternion_error_angle_deg(q):
    return 2*np.arccos(np.clip(abs(q[0]),0,1))*RAD2DEG
