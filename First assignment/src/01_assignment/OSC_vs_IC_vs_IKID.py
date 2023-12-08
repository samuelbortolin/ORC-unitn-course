import numpy as np
from numpy import nan
from numpy.linalg import inv, norm
from math import sqrt
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
import time, sys
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import OSC_vs_IC_vs_IKID_conf as conf

print("".center(conf.LINE_WIDTH,'#'))
print(" Manipulator: Impedence Control vs. Operational Space Control vs. Inverse Kinematics + Inverse Dynamics ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_TORQUES = 0
PLOT_EE_POS = 0
PLOT_EE_VEL = 0
PLOT_EE_ACC = 0

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

tests = []
tests += [{'controller': 'OSC', 'kp': 50,  'frequency': np.array([1.0, 1.0, 0.3])}]
tests += [{'controller': 'IC',  'kp': 50,  'frequency': np.array([1.0, 1.0, 0.3])}]
tests += [{'controller': 'IKID',  'kp': 50,  'frequency': np.array([1.0, 1.0, 0.3])}]
tests += [{'controller': 'OSC', 'kp': 100, 'frequency': np.array([1.0, 1.0, 0.3])}]
tests += [{'controller': 'IC',  'kp': 100, 'frequency': np.array([1.0, 1.0, 0.3])}]
tests += [{'controller': 'IKID',  'kp': 100, 'frequency': np.array([1.0, 1.0, 0.3])}]

tests += [{'controller': 'OSC', 'kp': 50,  'frequency': np.array([2.0, 2.0, 0.6])}]
tests += [{'controller': 'IC',  'kp': 50,  'frequency': np.array([2.0, 2.0, 0.6])}]
tests += [{'controller': 'IKID',  'kp': 50,  'frequency': np.array([2.0, 2.0, 0.6])}]
tests += [{'controller': 'OSC', 'kp': 100, 'frequency': np.array([2.0, 2.0, 0.6])}]
tests += [{'controller': 'IC',  'kp': 100, 'frequency': np.array([2.0, 2.0, 0.6])}]
tests += [{'controller': 'IKID',  'kp': 100, 'frequency': np.array([2.0, 2.0, 0.6])}]

plt.tight_layout(rect=[0, 0, 1, 0.95])
frame_id = robot.model.getFrameId(conf.frame_name)

simu = RobotSimulator(conf, robot)
tracking_err_osc = []   # list to contain the tracking error of OSC
tracking_err_ic  = []   # list to contain the tracking error of IC
tracking_err_ikid  = []   # list to contain the tracking error of IKID
if conf.use_viewer:
    simu.gui.addSphere('world/target', conf.REF_SPHERE_RADIUS, conf.REF_SPHERE_COLOR)
    simu.gui.addSphere("world/ee", conf.EE_SPHERE_RADIUS, conf.EE_SPHERE_COLOR)

for (test_id, test) in  enumerate(tests):
    description = str(test_id)+' Controller '+test['controller']+' kp='+\
                  str(test['kp'])+' frequency=[{},{},{}]'.format(test['frequency'][0],test['frequency'][1],test['frequency'][2])
    print(description)

    kp = test['kp']             # proportional gain of tracking task
    kd = 2*np.sqrt(kp)          # derivative gain of tracking task
    kp_j = 20.0                 # proportional gain of end effector task
    kd_j = 2*sqrt(kp_j)         # derivative gain of end effector task

    freq = test['frequency']
    simu.init(conf.q0)                              # initialize simulation state
    
    nx, ndx = 3, 3                          # size of x and its time derivative
    N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
    tau     = np.empty((robot.na, N))*nan    # joint torques
    tau_c   = np.empty((robot.na, N))*nan    # joint Coulomb torques
    q       = np.empty((robot.nq, N+1))*nan  # joint angles
    v       = np.empty((robot.nv, N+1))*nan  # joint velocities
    dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
    x       = np.empty((nx,  N))*nan        # end-effector position
    dx      = np.empty((ndx, N))*nan        # end-effector velocity
    ddx     = np.empty((ndx, N))*nan        # end effector acceleration
    x_ref   = np.empty((nx,  N))*nan        # end-effector reference position
    dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
    ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
    ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration
    ddq_des = np.empty((robot.nq, N+1))*nan        # joint desired acceleration
    
    two_pi_f             = 2*np.pi*freq   # frequency (time 2 PI)
    two_pi_f_amp         = np.multiply(two_pi_f, conf.amp)
    two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)
    
    t = 0.0
    PRINT_N = int(conf.PRINT_T/conf.dt)
    
    try:
        for i in range(0, N):
            time_start = time.time()
    
            # set reference trajectory
            x_ref[:,i]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
            dx_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
            ddx_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)

            # read current state from simulator
            v[:,i] = simu.v
            q[:,i] = simu.q

            # compute mass matrix M, bias terms h, gravity terms g
            robot.computeAllTerms(q[:,i], v[:,i])
            M = robot.mass(q[:,i], False)
            h = robot.nle(q[:,i], v[:,i], False)
            g = robot.gravity(q[:,i])

            J6 = robot.frameJacobian(q[:,i], frame_id, False)
            J = J6[:3,:]                                                                    # take first 3 rows of J6
            H = robot.framePlacement(q[:,i], frame_id, False)
            x[:,i] = H.translation                                                          # take the 3d position of the end-effector
            v_frame = robot.frameVelocity(q[:,i], v[:,i], frame_id, False)
            dx[:,i] = v_frame.linear                                                        # take linear part of 6d velocity
            #    dx[:,i] = J.dot(v[:,i])
            dJdq = robot.frameAcceleration(q[:,i], v[:,i], None, frame_id, False).linear

            # implement the components needed for your control laws here
            ddx_fb = kp * (x_ref[:,i] - x[:,i]) + kd*(dx_ref[:,i] - dx[:,i])        # Feedback acceleration
            ddx_des[:,i] = ddx_ref[:,i] + ddx_fb                                    # Desired acceleration
            Minv = inv(M)                                                           # M^-1
            J_Minv = J.dot(Minv)
            Lambda = inv(J_Minv.dot(J.T))

            # secondary task here
            J_T_pinv = Lambda.dot(J_Minv)                                           # Pseudo-inverse of J.T
            NJ = np.eye(robot.nv) - J.T.dot(J_T_pinv)                               # Null space of the pseudo-inverse of J.T
            J_moore = J.T.dot(inv(J.dot(J.T)))                                      # Moore Penrose pseudo-inverse of J.T
            NJ_moore = np.eye(robot.nv) - J_moore.dot(J)                            # Null space of the Moore Penrose pseudo-inverse of J
            ddq_pos_des = kp_j * (conf.q0 - q[:,i]) - kd_j*v[:,i]         # Let's choose ddq_pos_des to stabilize the initial joint configuration
            tau_0 = M.dot(ddq_pos_des)                                              # M*ddq_pos_des

            # desired joint acceleration for IKID here
            # ddq_des[:,i] = J_moore.dot(ddx_des[:,i] - dJdq)                         # without tau0
            ddq_des[:,i] = J_moore.dot(ddx_des[:,i] - dJdq) + NJ_moore.dot(ddq_pos_des) # with tau0

            # update sphere position
            if conf.use_viewer:
                simu.gui.applyConfiguration('world/target',x_ref[:,i].tolist()+[0.,0.,0.,1.])
                simu.gui.applyConfiguration("world/ee",x[:,i].tolist()+[0.,0.,0.,1.])             

            # define the control laws here
            if(test['controller']=='OSC'):      # Operational Space Control
                mu = Lambda.dot(J_Minv.dot(h) - dJdq)
                f = Lambda.dot(ddx_des[:,i]) + mu
                # tau[:,i] = J.T.dot(f)           # without tau0
                tau[:,i] = J.T.dot(f) + NJ.dot(tau_0 + h) # with tau0

            elif(test['controller']=='IC'):     # Impedence Control
                # tau[:,i] = h + J.T.dot(8*ddx_fb) # without tau0
                tau[:,i] = h + J.T.dot(8*ddx_fb) + NJ.dot(tau_0) # with tau0

            elif(test['controller']=='IKID'):   # Inverse Kinematics - Inverse Dynamics Control
                tau[:,i] = M.dot(ddq_des[:,i]) + h

            else:
                print('ERROR: Unknown controller', test['controller'])
                sys.exit(0)

            # send joint torques to simulator
            simu.simulate(tau[:,i], conf.dt, conf.ndt)
            tau_c[:,i] = simu.tau_c
            ddx[:,i] = J.dot(simu.dv) + dJdq
            t += conf.dt
    
            time_spent = time.time() - time_start
            if(conf.simulate_real_time and time_spent < conf.dt): 
                time.sleep(conf.dt-time_spent)    

        tracking_err = np.sum(norm(x_ref-x, axis=0))/N
        desc = test['controller']+' kp='+str(test['kp'])+' f=[{},{},{}]'.format(test['frequency'][0],test['frequency'][1],test['frequency'][2])
        if(test['controller']=='OSC'):        
            tracking_err_osc += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IC'):
            tracking_err_ic += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IKID'):
            tracking_err_ikid += [{'value': tracking_err, 'description': desc}]        
        else:
            print('ERROR: Unknown controller', test['controller'])

        print('Average tracking error %.3f m\n'%(tracking_err))

        # PLOT STUFF
        tt = np.arange(0.0, N*conf.dt, conf.dt)

        if(PLOT_EE_POS):    
            (f, ax) = plut.create_empty_figure(nx)
            ax = ax.reshape(nx)
            for i in range(nx):
                ax[i].plot(tt, x[i,:], label='x')
                ax[i].plot(tt, x_ref[i,:], '--', label='x ref')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel(r'x_'+str(i)+' [m]')
            leg = ax[0].legend()
            leg.get_frame().set_alpha(0.5)
            f.suptitle(description,y=1)

        if(PLOT_EE_VEL):    
            (f, ax) = plut.create_empty_figure(nx)
            ax = ax.reshape(nx)
            for i in range(nx):
                ax[i].plot(tt, dx[i,:], label='dx')
                ax[i].plot(tt, dx_ref[i,:], '--', label='dx ref')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel(r'dx_'+str(i)+' [m/s]')
            leg = ax[0].legend()
            leg.get_frame().set_alpha(0.5)
            f.suptitle(description,y=1)
   
        if(PLOT_EE_ACC):    
            (f, ax) = plut.create_empty_figure(nx)
            ax = ax.reshape(nx)
            for i in range(nx):
                ax[i].plot(tt, ddx[i,:], label='ddx')
                ax[i].plot(tt, ddx_ref[i,:], '--', label='ddx ref')
                ax[i].plot(tt, ddx_des[i,:], '-.', label='ddx des')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel(r'ddx_'+str(i)+' [m/s^2]')
            leg = ax[0].legend()
            leg.get_frame().set_alpha(0.5)
            f.suptitle(description,y=1)
     
        if(PLOT_TORQUES):    
            (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
            ax = ax.reshape(robot.nv)
            for i in range(robot.nv):
                ax[i].plot(tt, tau[i,:], label=r'$\tau$ '+str(i))
                ax[i].plot(tt, tau_c[i,:], label=r'$\tau_c$ '+str(i))
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('Torque [Nm]')
            leg = ax[0].legend()
            leg.get_frame().set_alpha(0.5)
            f.suptitle(description,y=1)

    except Exception:
        print('No solution\n')

(f, ax) = plut.create_empty_figure()
for (i,err) in enumerate(tracking_err_osc):
    ax.plot(i, err['value'], 's', markersize=20, label=err['description'])
for (i,err) in enumerate(tracking_err_ic):
    ax.plot(i, err['value'], 'o', markersize=20, label=err['description'])
for (i,err) in enumerate(tracking_err_ikid):
    ax.plot(i, err['value'], 'X', markersize=20, label=err['description'])
ax.set_xlabel('Test')
ax.set_ylabel('Mean tracking error [m]')
leg = ax.legend(prop={'size': 15})
leg.get_frame().set_alpha(0.5)

plt.show()
