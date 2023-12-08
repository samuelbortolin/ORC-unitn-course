import pinocchio as se3
import numpy as np
from numpy.linalg import norm
import os
import math
import gepetto.corbaserver
import time
import subprocess

class ContactPoint:
    def __init__(self, model, data, frame_name):
        self.model = model      # robot model
        self.data = data        # robot data
        self.frame_name = frame_name    # name of reference frame associated to this contact point
        self.frame_id = model.getFrameId(frame_name)    # id of the reference frame
        self.active = False         # True if this contact point is in contact
        
    def get_position(self):
        ''' Get the current position of this contact point 
        '''
        M = self.data.oMf[self.frame_id]
        return M.translation
        
class ContactSurface:
    def __init__(self, name, pos, normal, K, B, mu):
        self.name = name        # name of this contact surface
        self.x0 = pos           # position of a point of the surface
        self.normal = normal    # direction of the normal to the surface
        self.K = K              # stiffness of the surface material
        self.B = B              # damping of the surface material
        self.mu = mu            # friction coefficient of the surface
        self.bias = self.x0.dot(self.normal)
        
    def check_collision(self, p):
        ''' Check the collision of the given point
            with this contact surface. If the point is not
            inside this surface, then return False.
        '''
        normal_penetration = self.bias - p.dot(self.normal)
        if(normal_penetration < 0.0):
            return False # no penetration
        return True


class Contact:
    def __init__(self, model, data, frame_name, normal, K, B, mu):
        self.model = model      # model of the robot
        self.data = data        # data of the robot
        self.frame_name = frame_name    # name of the reference frame associated to this contact
        self.normal = normal    # direction normal to the contact surface
        self.K = K              # stiffness of the contact surface
        self.B = B              # damping of the contact surface
        self.mu = mu            # friction coefficient of the contact surface
        self.frame_id = model.getFrameId(frame_name)
        self.reset_contact_position()

    def reset_contact_position(self):
        # Initialize anchor point p0, that is the initial (0-load) position of the spring
        self.p0 = self.data.oMf[self.frame_id].translation.copy()
        self.in_contact = True

    def compute_force(self):
        M = self.data.oMf[self.frame_id] # Homogeneous matrix corresponding to contact frame
        self.p = M.translation          # get position of contact point
        delta_p = self.p0 - self.p      # distance between anchor point and contact point

        R = se3.SE3(M.rotation, 0*M.translation)    # same as M but with translation set to zero
        v_local = se3.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear   # convert velocity from local frame to world frame

        # Doubt: should I use classic or spatial acceleration here?!
        dJv_local = se3.getFrameAcceleration(self.model, self.data, self.frame_id)
        dJv_local.linear += np.cross(v_local.angular, v_local.linear, axis=0)
        dJv_world = (R.act(dJv_local)).linear

        # compute contact force using spring-damper law
        self.f = self.K.dot(delta_p) - self.B.dot(v_world)
        f_N = self.f.dot(self.normal)   # normal force
        f_T = self.f - f_N*self.normal  # tangential force
        f_T_norm = norm(f_T)            # norm of normal force
        if(f_T_norm > self.mu*f_N):
            # contact is slipping 
            t_dir = f_T / f_T_norm  # direction of tangential force
            # saturate force at the friction cone boundary
            self.f = f_N*self.normal + self.mu*f_N*t_dir
            
            # update anchor point so that f is inside friction cone
            delta_p0 = (f_T_norm - self.mu*f_N) / self.K[0,0]
            self.p0 -= t_dir*delta_p0
            
        self.v = v_world
        self.dJv = dJv_world
        return self.f

    def getJacobianWorldFrame(self):
        J6 = se3.getFrameJacobian(self.model, self.data, self.frame_id, se3.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = J6[:3,:]
        return self.J


def randomize_robot_model(model_old, sigma):
    from random import uniform
    model = model_old.copy()
    for (ine, ine_old) in zip(model.inertias, model_old.inertias):
        ine.mass *= 1.0 + uniform(-sigma, sigma)*1e-2
        ine.lever *= 1.0 + uniform(-sigma, sigma)*1e-2
        ine.inertia *= 1.0 + uniform(-sigma, sigma)*1e-2
        print("mass", ine_old.mass, " => ", ine.mass)
    return model


class RobotSimulator:

    # Class constructor
    def __init__(self, conf, robot):
        self.conf = conf
        self.robot = robot
        if(conf.randomize_robot_model):
            self.model = randomize_robot_model(robot.model, conf.model_variation)
        else:
            self.model = self.robot.model
        self.data = self.model.createData()
        self.t = 0.0                    # time
        self.f = np.zeros(0)            # Contact forces
        self.nv = nv = self.model.nv    # Dimension of joint velocities vector
        self.na = na = robot.na         # number of actuated joints
        # Matrix S used as filter of vetor of inputs U
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na)))

        self.contacts = []
        self.candidate_contact_points = [] # candidate contact points
        self.contact_surfaces = []
        
        self.DISPLAY_T = conf.DISPLAY_T     # refresh period for viewer
        self.display_counter = self.DISPLAY_T
        self.init(conf.q0, None, True)
        
        self.tau_c = np.zeros(na)   # Coulomb friction torque
        self.simulate_coulomb_friction = conf.simulate_coulomb_friction
        self.simulation_type = conf.simulation_type
        if(self.simulate_coulomb_friction):
            self.tau_coulomb_max = 1e-2*conf.tau_coulomb_max*self.model.effortLimit
        else:
            self.tau_coulomb_max = np.zeros(na)
        
        if(norm(self.tau_coulomb_max)==0.0):
            self.simulate_coulomb_friction = False

        # for gepetto viewer
        if(conf.use_viewer):
            try:
                prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(prompt[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
            except:
                pass
            gepetto.corbaserver.Client()
            self.robot.initViewer(loadModel=True)
            self.gui = self.robot_display.viewer.gui
            if(conf.show_floor):
                self.robot.viewer.gui.createSceneWithFloor('world')
                self.gui.setLightingMode('world/floor', 'OFF')
            self.robot.displayCollisions(False)
            self.robot.displayVisuals(True)
            self.robot.display(self.q)            
            self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)
            

    # Re-initialize the simulator
    def init(self, q0=None, v0=None, reset_contact_positions=False):
        self.first_iter = True

        if q0 is not None:
            self.q = q0.copy()
            
        if(v0 is None):
            self.v = np.zeros(self.robot.nv)
        else:
            self.v = v0.copy()
        self.dv = np.zeros(self.robot.nv)
        self.resize_contact_data(reset_contact_positions)
        
        
    def resize_contact_data(self, reset_contact_positions=False):
        self.nc = len(self.contacts)
        self.nk = 3*self.nc
        self.f = np.zeros(self.nk)
        self.Jc = np.zeros((self.nk, self.model.nv))
        self.K = np.zeros((self.nk, self.nk))
        self.B = np.zeros((self.nk, self.nk))
        self.p0 = np.zeros(self.nk)
        self.p = np.zeros(self.nk)
        self.dp = np.zeros(self.nk)
        self.dJv = np.zeros(self.nk)

        # reset contact position
        if(reset_contact_positions):
            se3.forwardKinematics(self.model, self.data, self.q)
            se3.updateFramePlacements(self.model, self.data)
            i = 0
            for c in self.contacts:
                c.reset_contact_position()
                self.p0[i:i+3] = c.p0
                i += 3

        self.compute_forces(compute_data=True)

    def add_candidate_contact_point(self, frame_name):
        self.candidate_contact_points += [ContactPoint(self.model, self.data, frame_name)]
        
    def add_contact_surface(self, name, pos, normal, K, B, mu):
        ''' Add a contact surface (i.e., a wall) located at "pos", with normal 
            outgoing direction "normal", 3d stiffness K, 3d damping B.
        '''
        self.contact_surfaces += [ContactSurface(name, pos, normal, K, B, mu)]
        self.gui.addFloor('world/'+name)
        self.gui.setLightingMode('world/'+name, 'OFF')
        z = np.array([0.,0.,1.])
        axis = np.cross(normal, z)
        if(np.linalg.norm(axis)>1e-6):
            angle = math.atan2(np.linalg.norm(axis), normal.dot(z))
            aa = se3.AngleAxis(angle, axis)
            H = se3.SE3(aa.matrix(), pos)
            self.gui.applyConfiguration('world/'+name, se3.se3ToXYZQUATtuple(H))
        else:
            self.gui.applyConfiguration('world/'+name, pos.tolist()+[0.,0.,0.,1.])
        
    # Adds a contact, resets all quantities
#    def add_contact(self, frame_name, normal, K, B, mu):
    def add_contact(self, contact_point, contact_surface):
        c = Contact(self.model, self.data, frame_name, normal, K, B, mu)
        self.contacts += [c]
        self.resize_contact_data()
        i = 0
        for c in self.contacts:
            self.K[i:i+3, i:i+3] = c.K
            self.B[i:i+3, i:i+3] = c.B
            self.p0[i:i+3] = c.p0
            i += 3
        self.D = np.hstack((-self.K, -self.B))
        return c
        
    def collision_detection(self):
        for s in self.contact_surfaces:
            for c in self.candidate_contact_points:
                p = c.get_position()
                if(s.check_collision(p)):
                    if(not c.active):
                        print("Collision detected between point", c.frame_name, " at ", p)
                        c.active = True
                        c.contact = self.add_contact(c.frame_name, s.normal, s.K, s.B, s.mu)
                else:
                    if(c.active):
                        print("Contact lost between point", c.frame_name, " at ", p)
                        c.active = False
                        self.contacts.remove(c.contact)
                        self.resize_contact_data()


    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:
            se3.forwardKinematics(self.model, self.data, self.q, self.v)
#            se3.computeAllTerms(self.model, self.data, self.q, self.v)            
            # Computes the placements of all the operational frames according to the current joint placement stored in data
            se3.updateFramePlacements(self.model, self.data)
#            se3.computeJointJacobians(self.model, self.data, self.q)
            self.collision_detection()
        i = 0
        for c in self.contacts:
            self.f[i:i+3] = c.compute_force()
            i += 3
        return self.f


    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt
            
        if self.first_iter:
            self.compute_forces()
            self.first_iter = False

        # dv  = se3.aba(robot.model,robot.data,q,v,tauq,ForceDict(self.forces,NB))
        # (Forces are directly in the world frame, and aba wants them in the end effector frame)
        se3.computeAllTerms(self.model, self.data, self.q, self.v)
        se3.updateFramePlacements(self.model, self.data)
        M = self.data.M
        h = self.data.nle
        i = 0
        self.collision_detection()
        self.compute_forces(False)
        for c in self.contacts:
            J = c.getJacobianWorldFrame()
            self.Jc[i:i+3, :] = J
            i += 3

        if(self.simulate_coulomb_friction and self.simulation_type=='timestepping'):
            # minimize kinetic energy using time stepping
            from quadprog import solve_qp
            '''
            Solve a strictly convex quadratic program
            
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b
            
            Input Parameters
            ----------
            G : array, shape=(n, n)
            a : array, shape=(n,)
            C : array, shape=(n, m) matrix defining the constraints
            b : array, shape=(m), default=None, vector defining the constraints
            meq : int, default=0
                the first meq constraints are treated as equality constraints,
                all further as inequality constraints
            factorized : bool, default=False
                If True, then we are passing :math:`R^{−1}` (where :math:`G = R^T R`)
                instead of the matrix G in the argument G.
            '''
            # M (v' - v) = dt*S^T*(tau - tau_c) - dt*h + dt*J^T*f
            # M v' = M*v + dt*(S^T*tau - h + J^T*f) - dt*S^T*tau_c
            # M v' = b + B*tau_c
            # v' = Minv*(b + B*tau_c)
            b = M.dot(self.v) + dt*(self.S.T.dot(u) - h + self.Jc.T.dot(self.f))
            B = - dt*self.S.T
            # Minimize kinetic energy:
            # min v'.T * M * v'
            # min  (b+B*tau_c​).T*Minv*(b+B*tau_c​) 
            # min tau_c.T * B.T*Minv*B* tau_C + 2*b.T*Minv*B*tau_c
            Minv = np.linalg.inv(M)
            G = B.T.dot(Minv.dot(B))
            a = -b.T.dot(Minv.dot(B))
            C = np.vstack((np.eye(self.na), -np.eye(self.na)))
            c = np.concatenate((-self.tau_coulomb_max, -self.tau_coulomb_max))
            solution = solve_qp(G, a, C.T, c, 0)
            self.tau_c = solution[0]
#            print('Time %.3f'%self.t)
#            print('v : ', self.v.T)
            self.v = Minv.dot(b + B.dot(self.tau_c))
#            print("v': ", self.v.T)
#            print("tau_c:     ", self.tau_c.T)
#            print("tau_c unc: ", (b.T/dt))
#            print("tau_c_max: ", self.tau_coulomb_max.T)
            self.q = se3.integrate(self.model, self.q, self.v*dt)
        elif(self.simulation_type=='euler' or self.simulate_coulomb_friction==False):
            if(self.simulate_coulomb_friction):
                self.tau_c = self.tau_coulomb_max*np.sign(self.v[-self.na:])
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))  # use last forces
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q = se3.integrate(self.model, self.q, v_mean*dt)
        else:
            print("[ERROR] Unknown simulation type:", self.simulation_type)

#        self.compute_forces()
#        self.compute_forces(False)
        self.t += dt
        return self.q, self.v

    def reset(self):
        self.first_iter = True

    def simulate(self, u, dt=0.001, ndt=1):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        tau_c_avg = 0*self.tau_c
        for i in range(ndt):
            self.q, self.v = self.step(u, dt/ndt)
            tau_c_avg += self.tau_c
        self.tau_c = tau_c_avg / ndt

        if(self.conf.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0.0:
                self.robot_display.display(self.q)
                self.display_counter = self.DISPLAY_T

        return self.q, self.v, self.f
