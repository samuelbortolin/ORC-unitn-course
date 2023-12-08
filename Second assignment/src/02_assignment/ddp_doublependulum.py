# -*- coding: utf-8 -*-

import numpy as np
from ddp import DDPSolver
import pinocchio as pin

class DDPSolverLinearDyn(DDPSolver):
    ''' The linear system dynamics are defined by:
            x_{t+1} = A x_t + B u_t
        The task is defined by a quadratic cost: sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, DEBUG=False):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.underact = underact
        self.dt = dt
        self.nx = h_x.shape[1]
        self.nu = self.nx
        
    def cost(self, X, U):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        N = U.shape[0]
        cost = self.cost_final(X[-1,:])
        for i in range(N):
            cost += self.cost_running(i, X[i,:], U[i,:])
        return cost
        
    def cost_running(self, i, x, u):                                    
        ''' Running cost at time step i for state x and control u '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[i,:,:], x)) \
                + np.dot(self.h_x[i,:].T, x) + self.h_s[i] \
                + 0.5*self.lmbda*np.dot(u.T, u) + self.underact * np.dot(u.T, np.dot(np.array([[0, 0], [0, 1]]), u))               # Add term penalizing the torque provided by the second motor with weight self.underact  
        return cost
        
    def cost_final(self, x):
        ''' Final cost for state x '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[-1,:,:], x)) \
                + np.dot(self.h_x[-1,:].T, x) + self.h_s[-1]
        return cost
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        c_x = self.h_x[i,:] + np.dot(self.H_xx[i,:,:], x)
        return c_x
        
    def cost_final_x(self, x):
        ''' Gradient of the final cost w.r.t. x '''
        c_x = self.h_x[-1,:] + np.dot(self.H_xx[-1,:,:], x)
        return c_x
        
    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        c_u = self.lmbda * u + self.underact * 2 * np.dot(np.array([[0, 0], [0, 1]]), u)
        return c_u
        
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        return self.H_xx[i,:,:]
        
    def cost_final_xx(self, x):
        ''' Hessian of the final cost w.r.t. x '''
        return self.H_xx[-1,:,:]
        
    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        return self.lmbda * np.eye(self.nu) + self.underact * 2 * np.array([[0, 0], [0, 1]])                              # Modify accordingly the hessian of the running cost w.r.t. u
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        return np.zeros((self.nx, self.nu))

class DDPSolverDoublePendulum(DDPSolverLinearDyn):
    ''' 
        Derived class of DDPSolverLinearDyn implementing the multi-body dynamics of a double pendulum.
        The task is defined by a quadratic cost: sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, robot, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, DEBUG=False, simu=None):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.robot = robot
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.underact = underact
        self.nx = h_x.shape[1]
        self.nu = robot.na
        self.dt = dt
        self.simu = simu
        
        nv = self.robot.nv # number of joints
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)

    ''' System dynamics '''
    def f(self, x, u):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        if SELECTION_MATRIX == 1:
            self.S = np.array([[1, 0], [0, 0]])            # Selection matrix
            u = np.dot(self.S, u)                          # When using the selection matrix method, the correct dynamics must be used in the forward pass of DDP.
                                                           # Thus, 0 torque must be sent to the joint not actuated, which means u = S*u
        ddq = pin.aba(model, data, q, v, u)
        self.dx[:nv] = v
        self.dx[nv:] = ddq
        return x + self.dt * self.dx
           
    def f_x_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
    def f_u_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up)
            Fu[:,i] = (fp-f0)/delta
                
        return Fu
        
    def f_x(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
                
        # first compute Jacobians for continuous time dynamics
        pin.computeABADerivatives(model, data, q, v, u)
        self.Fx[:nv, :nv] = 0.0
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fx[nv:, :nv] = data.ddq_dq
        self.Fx[nv:, nv:] = data.ddq_dv

        if SELECTION_MATRIX==1 and ACTUATION_PENALTY==0:
            self.Fu[nv:, :] = np.dot(data.Minv, self.S)                                      # Partial derivatives of system dynamics w.r.t. u    
        elif SELECTION_MATRIX==0 and ACTUATION_PENALTY==1:
            self.Fu[nv:, :] = data.Minv
        else:
            raise RuntimeError("No method has been chosen to consider the underactuated case")    

        # Convert them to discrete time
        self.Fx = np.identity(2*nv) + dt * self.Fx
        self.Fu *= dt
        
        return self.Fx
    
    def f_u(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. u '''
        return self.Fu
        
    def callback(self, X, U):
        pass
        for i in range(0, N):
            time_start = time.time()
            self.simu.display(X[i,:self.robot.nq])
            time_spent = time.time() - time_start
            if(time_spent < self.dt):
                time.sleep(self.dt-time_spent)
                
    def start_simu(self, X, U, KK, dt_sim):
        t = 0.0
        simu = self.simu
        simu.init(X[0,:self.robot.nq])
        ratio = int(self.dt/dt_sim)
        N_sim = N * ratio
        n = X.shape[1]
        m = U.shape[1]
        X_sim = np.zeros((N+1, n))
        U_sim = np.zeros((N, m))
        X[0,:] = x0
        tau_arr = np.zeros((m,ratio))
        x_arr = np.zeros((n,ratio))
        j2 = 0
        i2 = 0
        print("Start simulation")
        time.sleep(1)
        for i in range(0, N_sim):
            time_start = time.time()
    
            # compute the index corresponding to the DDP time step
            j = int(np.floor(i/ratio))
            # compute joint torques
            x = np.hstack([simu.q, simu.v])
            tau = U[j,:] + KK[j,:,:] @ (X[j,:] - x)
            tau[1] = 0.0                                                        # No motor on the second joint
            if j2!=j:
                i2=0
            tau_arr[:,i2] = tau
            # send joint torques to simulator
            q,v,f = simu.simulate(tau, dt_sim)
            x_arr[:,i2] = np.hstack([q,v])
            if j2!=j:
                U_sim[j2,:] = np.array([np.mean(tau_arr[0]),np.mean(tau_arr[1])])                           
                X_sim[j2+1,:] = np.array([np.mean(x_arr[0]),np.mean(x_arr[1]),np.mean(x_arr[2]),np.mean(x_arr[3])])
                j2=j
            i2+=1    
            t += dt_sim
            time_spent = time.time() - time_start
            if(time_spent < dt_sim):
                time.sleep(dt_sim-time_spent)
        print("Simulation finished")

        # compute cost of each task 
        cost = self.cost(X_sim, U_sim)
        print("Cost Sim.", cost)
        print("Effort Sim.", np.linalg.norm(U_sim))

        return X_sim, U_sim

        
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import time
    from example_robot_data.robots_loader import load
    from orc.utils.robot_wrapper import RobotWrapper
    from orc.utils.robot_simulator import RobotSimulator
    import ddp_doublependulum_conf as conf
    np.set_printoptions(precision=3, suppress=True)
    
    ''' Test DDP with a double pendulum
    '''
    print("".center(conf.LINE_WIDTH,'#'))
    print(" DDP - Double Pendulum ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')

    N = conf.N               # horizon size
    dt = conf.dt             # control time step
    mu = 1e-4                # initial regularization
    ddp_params = {}
    ddp_params['alpha_factor'] = 0.5
    ddp_params['mu_factor'] = 10.
    ddp_params['mu_max'] = 1e0
    ddp_params['min_alpha_to_increase_mu'] = 0.1
    ddp_params['min_cost_impr'] = 1e-1
    ddp_params['max_line_search_iter'] = 10
    ddp_params['exp_improvement_threshold'] = 1e-3
    ddp_params['max_iter'] = 50
    DEBUG = False

    SELECTION_MATRIX = conf.SELECTION_MATRIX       
    ACTUATION_PENALTY = conf.ACTUATION_PENALTY       

    r = load("double_pendulum")
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    
    n = nq+nv                                                   # state size
    m = robot.na                                                # control size
    U_bar = np.zeros((N,m));                                    # initial guess for control inputs
    x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))          # initial state
    x_tasks = np.concatenate((conf.qT, np.zeros(robot.nv)))     # goal state
    N_task = N;                                                 # time step to reach goal state

    PLOT_TORQUES = 1                                            # flag to plot the control torques
    PLOT_POS = 1                                                # flag to plot the control torques
    
    tau_g = robot.nle(conf.q0, np.zeros(robot.nv))
    for i in range(N):
        U_bar[i,:] = tau_g
    
    ''' TASK FUNCTION  '''
    lmbda = 1e1                                                     # control regularization
    if SELECTION_MATRIX==1:
        underact = 0                                                # underactuation penalty weight
    elif ACTUATION_PENALTY==1:
        underact = 1e5
    H_xx = np.zeros((N+1, n, n))
    h_x  = np.zeros((N+1, n))
    h_s  = np.zeros(N+1)
    W = np.diagflat(np.concatenate([np.ones(nq), np.zeros(nv)]))
    for i in range(N_task):
        H_xx[i,:,:]  = W
        h_x[i,:]     = -W @ x_tasks
        h_s[i]       = 0.5*x_tasks.T @ W @ x_tasks
    
    print("Displaying desired goal configuration")
    simu = RobotSimulator(conf, robot)
    simu.display(conf.qT)
    time.sleep(1.)
    
    solver = DDPSolverDoublePendulum("doublependulum", robot, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, DEBUG, simu)
    
    (X,U,KK) = solver.solve(x0, U_bar, mu)
    if SELECTION_MATRIX==1:    
        print("\nMETHOD = SELECTION MATRIX")
    elif ACTUATION_PENALTY==1:
        print("\nMETHOD = ACTUATION_PENALTY")
    solver.print_statistics(x0, U, KK, X)
    
    print("Show reference motion")
    for i in range(0, N):
        time_start = time.time()
        simu.display(X[i,:nq])
        time_spent = time.time() - time_start
        if(time_spent < dt):
            time.sleep(dt-time_spent)
    print("Reference motion finished")
    time.sleep(1)
    
    print("Show real simulation")
    x_sim, tau_sim = solver.start_simu(X, U, KK, conf.dt_sim)

    tt = np.arange(0.0, N*conf.dt, conf.dt)
    if PLOT_TORQUES:    
        fig, ax = plt.subplots(2,1)
        ax[0].plot(tt, tau_sim[:,0], label=r'Simulated torque 1st joint')
        ax[0].plot(tt, U[:,0], label=r'Reference torque 1st joint')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Torque [Nm]')
        ax[1].plot(tt, tau_sim[:,1], label=r'Simulated torque 2nd joint')
        ax[1].plot(tt, U[:,1], label=r'Reference torque 2nd joint ')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Torque [Nm]')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    if PLOT_POS:    
        fig, ax = plt.subplots(2,1)
        ax = ax.reshape(robot.nq)
        ax[0].plot(tt, x_sim[:-1,0], label=r'Simulated position 1st joint')
        ax[0].plot(tt, X[:-1,0], label=r'Reference position 1st joint')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('[rad]')
        ax[1].plot(tt, x_sim[:-1,1], label=r'Simulated position 2nd joint')
        ax[1].plot(tt, X[:-1,1], label=r'Reference position 2nd joint')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('[rad]')
        ax[0].legend()
        ax[1].legend()
        plt.show()
