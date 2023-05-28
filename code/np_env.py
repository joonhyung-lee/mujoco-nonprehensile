import numpy as np
from util import r2rpy

class NonPrehensileMarkovDecisionProcessClass():
    """
        UR5e Non-prehensile MDP Class from MuJoCo Env
    """
    def __init__(self,env,HZ=50,history_total_sec=2.0,history_intv_sec=0.1,VERBOSE=True):
        """
            Initialize
        """
        self.env               = env # MuJoCo environment
        self.HZ                = HZ
        self.dt                = 1/self.HZ
        self.history_total_sec = history_total_sec # history in seconds
        self.n_history         = int(self.HZ*self.history_total_sec) # number of history
        self.history_intv_sec  = history_intv_sec
        self.history_intv_tick = int(self.HZ*self.history_intv_sec) # interval between state in history
        self.history_ticks     = np.arange(0,self.n_history,self.history_intv_tick)
        
        self.mujoco_nstep      = self.env.HZ // self.HZ # nstep for MuJoCo step
        self.VERBOSE           = VERBOSE
        self.tick              = 0
        
        self.name              = env.name
        self.state_prev        = self.get_state()
        self.action_prev       = self.sample_action()
        
        # Dimensions
        self.state_dim         = len(self.get_state())
        self.state_history     = np.zeros((self.n_history,self.state_dim))
        self.tick_history      = np.zeros((self.n_history,1))
        self.o_dim             = len(self.get_observation())
        self.a_dim             = env.n_ctrl
        
        if VERBOSE:
            print ("[%s] Instantiated"%
                   (self.name))
            print ("   [info] dt:[%.4f] HZ:[%d], env-HZ:[%d], mujoco_nstep:[%d], state_dim:[%d], o_dim:[%d], a_dim:[%d]"%
                   (self.dt,self.HZ,self.env.HZ,self.mujoco_nstep,self.state_dim,self.o_dim,self.a_dim))
            print ("   [history] total_sec:[%.2f]sec, n:[%d], intv_sec:[%.2f]sec, intv_tick:[%d]"%
                   (self.history_total_sec,self.n_history,self.history_intv_sec,self.history_intv_tick))
            print ("   [history] ticks:%s"%(self.history_ticks))
        
    def get_state(self):
        """
            Get state (8)
            : Current state consists of 
                1) current joint position (6)
                2) current joint velocity (6)
                3) eef position (3)
                4) torso rotation (9)
                6) contact info (N)
        """
        # Joint position
        qpos = self.env.data.qpos[self.env.ctrl_joint_idxs] # joint position
        # Joint velocity
        qvel = self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        # eef position
        p_tcp = self.env.get_p_body(body_name='tcp_link')
        # eef rotation
        R_tcp = self.env.get_R_body(body_name='tcp_link').flatten()
        # Contact information
        contact_info = np.zeros(self.env.n_sensor)
        contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
        contact_info[contact_idxs] = 1.0 # 1 means contact occurred
        # Concatenate information
        state = np.concatenate([
            qpos,
            qvel/10.0, # scale
            p_tcp,
            R_tcp,
            contact_info
        ])
        return state
    
    def get_observation(self):
        """
            Get observation 
        """
        
        # Sparsely accumulated history vector 
        state_history_sparse = self.state_history[self.history_ticks,:]
        
        # Concatenate information
        obs = np.concatenate([
            state_history_sparse
        ])
        return obs.flatten()

    def sample_action(self):
        """
            Sample action (8)
        """
        a_min  = self.env.ctrl_ranges[:,0]
        a_max  = self.env.ctrl_ranges[:,1]
        action = a_min + (a_max-a_min)*np.random.rand(len(a_min))
        return action
        
    def step(self,a,max_time=np.inf):
        """
            Step forward
        """
        # Increse tick
        self.tick = self.tick + 1
        
        # Previous eef position and yaw angle in degree
        p_eef_prev         = self.env.get_p_body('tcp_link')

        # Run simulation for 'mujoco_nstep' steps
        self.env.step(ctrl=a,nstep=self.mujoco_nstep)
        
        # Current eef position and yaw angle in degree
        p_eef_curr         = self.env.get_p_body('tcp_link')

        # Compute the done signal
        if (self.get_sim_time() >= max_time):
            d = True
        else:
            d = False
        
        # Compute forward reward
        x_diff = p_eef_curr[0] - p_eef_prev[0] # x-directional displacement
        r_forward = x_diff/self.dt
        
        # Check self-collision (excluding 'floor')

        # Compute reward
        r = np.array(r_forward)
        
        # Accumulate state history (update 'state_history')
        self.accumulate_state_history()
        
        # Next observation 'accumulate_state_history' should be called before calling 'get_observation'
        o_prime = self.get_observation()
        
        # Other information
        info = {'p_eef_prev':p_eef_prev,'p_eef_curr':p_eef_curr,
                'x_diff':x_diff,'r_forward':r_forward}
        
        # Return
        return o_prime,r,d,info
    
    def render(self,TRACK_EEF=True,PLOT_WORLD_COORD=True,PLOT_EEF_COORD=True,
               PLOT_SENSOR=True,PLOT_CONTACT=True,PLOT_TIME=True):
        """
            Render
        """
        # Change lookat
        if TRACK_EEF:
            p_lookat = self.env.get_p_body('tcp_link')
            self.env.update_viewer(lookat=p_lookat,CALL_MUJOCO_FUNC=False)
        # World coordinate
        if PLOT_WORLD_COORD:
            self.env.plot_T(p=np.zeros(3),R=np.eye(3,3),PLOT_AXIS=True,axis_len=1.0,axis_width=0.0025)
        # Plot snapbot base
        if PLOT_EEF_COORD:
            p_tcp,R_tcp = self.env.get_pR_body(body_name='tcp_link') # update viewer
            self.env.plot_T(p=p_tcp,R=R_tcp,PLOT_AXIS=True,axis_len=0.25,axis_width=0.0025)
        # Plot contact sensors
        if PLOT_SENSOR:
            contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
            for idx in contact_idxs:
                sensor_name = self.env.sensor_names[idx]
                p_sensor = self.env.get_p_sensor(sensor_name)
                self.env.plot_sphere(p=p_sensor,r=0.02,rgba=[1,0,0,0.2])
        # Plot contact info
        if PLOT_CONTACT:
            p_contacts,f_contacts,geom1s,geom2s = self.env.get_contact_info()
            for (p_contact,f_contact,geom1,geom2) in zip(p_contacts,f_contacts,geom1s,geom2s):
                f_norm = np.linalg.norm(f_contact)
                f_uv   = f_contact / (f_norm+1e-8)
                r_stem = 0.01
                f_len  = 0.2 # f_norm*0.05
                label  = '' #'[%s]-[%s]'%(geom1,geom2)
                self.env.plot_arrow(
                    p=p_contact,uv=f_uv,r_stem=0.01,len_arrow=f_len,rgba=[1,0,0,0.4],label='')
                self.env.plot_arrow(
                    p=p_contact,uv=-f_uv,r_stem=0.01,len_arrow=f_len,rgba=[1,0,0,0.4],label='')
                self.env.plot_sphere(p=p_contact,r=0.0001,label=label)
        # Plot time and tick on top of eef
        if PLOT_TIME:
            self.env.plot_T(p=p_tcp+0.25*R_tcp[:,2],R=np.eye(3,3),
                       PLOT_AXIS=False,label='[%.2f]sec'%(self.env.get_sim_time()))
        # Do render
        self.env.render()

    def reset(self):
        """
            Reset
        """
        # Reset parameters
        self.tick = 0
        # Reset env
        self.env.reset()
        # Reset history
        self.state_history = np.zeros((self.n_history,self.state_dim))
        self.tick_history  = np.zeros((self.n_history,1))
        # Get observation
        o = self.get_observation()
        return o
        
    def init_viewer(self):
        """
            Initialize viewer
        """
        self.env.init_viewer(
            viewer_title='%s'%(self.name),viewer_width=1200,viewer_height=800,
            viewer_hide_menus=True)
        self.env.update_viewer(
            azimuth=95.0,distance=1.00,elevation=-27,lookat=[0.1,0.05,0.16],
            VIS_TRANSPARENT=False,VIS_CONTACTPOINT=True,
            contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
            VIS_JOINT=True,jointlength=0.5,jointwidth=0.1,jointrgba=[0.2,0.6,0.8,0.6])
        
    def close_viewer(self):
        """
            Close viewer
        """
        self.env.close_viewer()
        
    def get_sim_time(self):
        """
            Get time (sec)
        """
        return self.env.get_sim_time()
    
    def is_viewer_alive(self):
        """
            Check whether the viewer is alive
        """
        return self.env.is_viewer_alive()
    
    def accumulate_state_history(self):
        """
            Get state history
        """
        state = self.get_state()
        # Accumulate 'state' and 'tick'
        self.state_history[1:,:] = self.state_history[:-1,:]
        self.state_history[0,:]  = state
        self.tick_history[1:,:]  = self.tick_history[:-1,:]
        self.tick_history[0,:]   = self.tick
        
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.env.viewer_pause()
        
    def grab_image(self,resize_rate=1.0,interpolation=0):
        """
            Grab image
        """
        return self.env.grab_image(resize_rate=resize_rate,interpolation=interpolation)
