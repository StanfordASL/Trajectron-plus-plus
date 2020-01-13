import numpy as np
from pyquaternion import Quaternion
from . import ActuatorAngle
from scipy.interpolate import splrep, splev, CubicSpline
from scipy.integrate import cumtrapz


class Node(object):
    def __init__(self, type, position=None, velocity=None, acceleration=None, heading=None, orientation=None,
                 length=None, width=None, height=None, first_timestep=0, is_robot=False):
        self.type = type
        self.position = position
        self.heading = heading
        self.length = length
        self.wdith = width
        self.height = height
        self.orientation = orientation
        self.velocity = velocity
        self.acceleration = acceleration
        self.first_timestep = first_timestep
        self.dimensions = ['x', 'y', 'z']
        self.is_robot = is_robot
        self._last_timestep = None
        self.description = ""

    def __repr__(self):
        return self.type.name

    def scene_ts_to_node_ts(self, scene_ts):
        """
        Transforms timestamp from scene into timeframe of node data.
        :param scene_ts: Scene timesteps
        :return: ts: Transformed timesteps, paddingl: Number of timesteps in scene range which are not available in
                    node data before data is available. paddingu: Number of timesteps in scene range which are not
                    available in node data after data is available.
        """
        paddingl = (self.first_timestep - scene_ts[0]).clip(0)
        paddingu = (scene_ts[1] - self.last_timestep).clip(0)
        ts = np.array(scene_ts).clip(min=self.first_timestep, max=self.last_timestep) - self.first_timestep
        return ts, paddingl, paddingu

    def history_points_at(self, ts):
        """
        Number of history points in trajectory. Timestep is exclusive.
        :param ts: Scene timestep where the number of history points are queried.
        :return: Number of history timesteps.
        """
        return ts - self.first_timestep

    def get_entity(self, ts_scene, entity, dims, padding=np.nan):
        if ts_scene.size == 1:
            ts_scene = np.array([ts_scene, ts_scene])
        length = ts_scene[1] - ts_scene[0] + 1 # ts is inclusive
        entity_array = np.zeros((length, len(dims))) * padding
        ts, paddingl, paddingu = self.scene_ts_to_node_ts(ts_scene)
        entity_array[paddingl:length - paddingu] = np.array([getattr(getattr(self, entity), d)[ts[0]:ts[1]+1] for d in dims]).T
        return entity_array

    def get(self, ts_scene, state, padding=np.nan):
        return np.hstack([self.get_entity(ts_scene, entity, dims, padding) for entity, dims in state.items()])


    @property
    def timesteps(self):
        return self.position.x.size

    @property
    def last_timestep(self):
        if self._last_timestep is None:
            self._last_timestep = self.first_timestep + self.timesteps - 1
        return self._last_timestep


class BicycleNode(Node):
    def __init__(self, type, position=None, velocity=None, acceleration=None, heading=None, orientation=None,
                 length=None, width=None, height=None, first_timestep=0, actuator_angle=None):
        super().__init__(type, position=position, velocity=velocity, acceleration=acceleration, heading=heading,
                         orientation=orientation, length=length, width=width, height=height,
                         first_timestep=first_timestep)
        self.actuator_angle = actuator_angle

    # TODO Probably wrong. Differential of magnitude is not euqal to the  the magnitude of the differentials
    def calculate_steering_angle_old(self, vel_tresh=0.0):
        vel = np.linalg.norm(np.hstack((np.expand_dims(self.velocity.x, 1), np.expand_dims(self.velocity.y, 1))), axis=1)

        beta = np.arctan2(self.velocity.y, self.velocity.x) - self.heading.value
        beta[vel < vel_tresh] = 0.
        steering_angle = np.arctan2(2 * np.sin(beta), np.cos(beta))
        steering_angle[np.abs(steering_angle) > np.pi / 2] = 0 # Velocity Outlier

        aa = ActuatorAngle()
        aa.steering_angle = np.zeros_like(np.arctan2(2 * np.sin(beta), np.cos(beta)))
        self.actuator_angle = aa

    def calculate_steering_angle(self, dt, steering_tresh=0.0, vel_tresh=0.0):
        t = np.arange(0, self.timesteps * dt, dt)
        s = 0.01 * len(t)
        #c_pos_x_g_x_tck = CubicSpline(t, np.array(pos_x_filtert))
        #c_pos_y_g_x_tck = CubicSpline(t, np.array(pos_y_filtert))
        #c_pos_x_g_x_tck = splrep(t, self.position.x, s=s)
        #c_pos_y_g_x_tck = splrep(t, self.position.y, s=s)

        #vel_x_g = c_pos_x_g_x_tck(t, 1)
        #vel_y_g = c_pos_y_g_x_tck(t, 1)
        #vel_x_g = splev(t, c_pos_x_g_x_tck, der=1)
        #vel_y_g = splev(t, c_pos_y_g_x_tck, der=1)

        vel_x_g = self.velocity.x
        vel_y_g = self.velocity.y

        v_x_ego = []
        h = []
        for t in range(self.timesteps):
            dh_max = 1.2 / self.length * (np.linalg.norm(np.array([vel_x_g[t], vel_y_g[t]])))
            heading = np.arctan2(vel_y_g[t], vel_x_g[t])
            #if len(h) > 0 and np.abs(heading - h[-1]) > dh_max:
            #    heading = h[-1]
            h.append(heading)
            q = Quaternion(axis=(0.0, 0.0, 1.0), radians=heading)
            v_x_ego_t = q.inverse.rotate(np.array([vel_x_g[t], vel_y_g[t], 1]))[0]
            if v_x_ego_t < 0.0:
                v_x_ego_t = 0.
            v_x_ego.append(v_x_ego_t)

        v_x_ego = np.stack(v_x_ego, axis=0)
        h = np.stack(h, axis=0)

        dh = np.gradient(h, dt)

        sa = np.arctan2(dh * self.length, v_x_ego)
        sa[(dh == 0.) | (v_x_ego == 0.)] = 0.
        sa = sa.clip(min=-steering_tresh, max=steering_tresh)

        a = np.gradient(v_x_ego, dt)
        # int = self.integrate_bicycle_model(np.array([a]),
        #                                    sa,
        #                                    np.array([h[0]]),
        #                                    np.array([self.position.x[0],
        #                                              self.position.y[0]]),
        #                                    v_x_ego[0],
        #                                    self.length, 0.5)
        # p = np.stack((self.position.x, self.position.y), axis=1)
        #
        # #assert ((int[0] - p) < 1.0).all()

        aa = ActuatorAngle()
        aa.steering_angle = sa
        self.acceleration.m = a
        self.actuator_angle = aa

    def inverse_np_gradient(self, f, dx, F0=0.):
        N = f.shape[0]
        l = f.shape[-1]
        l2 = np.ceil(l / 2).astype(int)
        return (F0 +
                ((2 * dx) *
                 np.c_['-1',
                       np.r_['-1', np.zeros((N, 1)), f[..., 1:-1:2].cumsum(axis=-1)],
                       f[..., ::2].cumsum(axis=-1) - f[..., [0]] / 2]
                 ).reshape((N, 2, l2)).reshape(N, 2 * l2, order='F')[:, :l]
                )

    def integrate_trajectory(self, v, x0, dt):
        xd_ = self.inverse_np_gradient(v[..., 0], dx=dt, F0=x0[0])
        yd_ = self.inverse_np_gradient(v[..., 1], dx=dt, F0=x0[1])
        integrated = np.stack([xd_, yd_], axis=2)
        return integrated

    def integrate_bicycle_model(self, a, sa, h0, x0, v0, l, dt):
        v_m = self.inverse_np_gradient(a, dx=0.5, F0=v0)

        dh = (np.tan(sa) / l) * v_m[0]
        h = self.inverse_np_gradient(np.array([dh]), dx=dt, F0=h0)

        vx = np.cos(h) * v_m
        vy = np.sin(h) * v_m

        v = np.stack((vx, vy), axis=2)
        return self.integrate_trajectory(v, x0, dt)

    def calculate_steering_angle_keep(self, dt, steering_tresh=0.0, vel_tresh=0.0):

        vel_approx = np.linalg.norm(np.stack((self.velocity.x, self.velocity.y), axis=0), axis=0)
        mask = np.ones_like(vel_approx)
        mask[vel_approx < vel_tresh] = 0

        t = np.arange(0, self.timesteps * dt, dt)
        pos_x_filtert = []
        pos_y_filtert = []
        s = None
        for i in range(mask.size):
            if mask[i] == 0 and s is None:
                s = i
            elif mask[i] != 0 and s is not None:
                t_start = t[s-1]
                pos_x_start = self.position.x[s-1]
                pos_y_start = self.position.y[s-1]
                t_mean = t[s:i].mean()
                pos_x_mean = self.position.x[s:i].mean()
                pos_y_mean = self.position.y[s:i].mean()
                t_end = t[i]
                pos_x_end = self.position.x[i]
                pos_y_end = self.position.y[i]
                for step in range(s, i+1):
                    if t[step] <= t_mean:
                        pos_x_filtert.append(pos_x_start + ((t[step] - t_start) / (t_mean - t_start)) * (pos_x_mean - pos_x_start))
                        pos_y_filtert.append(pos_y_start + ((t[step] - t_start) / (t_mean - t_start)) * (pos_y_mean - pos_y_start))
                    else:
                        pos_x_filtert.append(pos_x_mean + ((t[step] - t_end) / (t_end - t_mean)) * (pos_x_end - pos_x_mean))
                        pos_y_filtert.append(pos_y_mean + ((t[step] - t_end) / (t_end - t_mean)) * (pos_y_end - pos_y_mean))
                s = None
            elif mask[i] != 0 and s is None:
                pos_x_filtert.append(self.position.x[i].mean())
                pos_y_filtert.append(self.position.y[i].mean())
        if s is not None:
            t_start = t[s - 1]
            pos_x_start = self.position.x[s - 1]
            pos_y_start = self.position.y[s - 1]
            t_mean = t[s:i].max()
            pos_x_mean = self.position.x[s:i].mean()
            pos_y_mean = self.position.y[s:i].mean()
            for step in range(s, i+1):
                pos_x_filtert.append(
                    pos_x_start + ((t[step] - t_start) / (t_mean - t_start)) * (pos_x_mean - pos_x_start))
                pos_y_filtert.append(
                    pos_y_start + ((t[step] - t_start) / (t_mean - t_start)) * (pos_y_mean - pos_y_start))

        s = 0.001 * len(t)
        #c_pos_x_g_x_tck = CubicSpline(t, np.array(pos_x_filtert))
        #c_pos_y_g_x_tck = CubicSpline(t, np.array(pos_y_filtert))
        c_pos_x_g_x_tck = splrep(t, np.array(pos_x_filtert), s=s)
        c_pos_y_g_x_tck = splrep(t, np.array(pos_y_filtert), s=s)

        #vel_x_g = c_pos_x_g_x_tck(t, 1)
        #vel_y_g = c_pos_y_g_x_tck(t, 1)
        vel_x_g = splev(t, c_pos_x_g_x_tck, der=1)
        vel_y_g = splev(t, c_pos_y_g_x_tck, der=1)

        v_x_ego = []
        h = []
        for t in range(self.timesteps):
            dh_max = 1.19 / self.length * (np.linalg.norm(np.array([vel_x_g[t], vel_y_g[t]])))
            heading = np.arctan2(vel_y_g[t], vel_x_g[t])
            if len(h) > 0 and np.abs(heading - h[-1]) > dh_max:
                heading = h[-1]
            h.append(heading)
            q = Quaternion(axis=(0.0, 0.0, 1.0), radians=heading)
            v_x_ego_t = q.inverse.rotate(np.array([vel_x_g[t], vel_y_g[t], 1]))[0]
            if v_x_ego_t < 0.0:
                v_x_ego_t = 0.
            v_x_ego.append(v_x_ego_t)

        v_x_ego = np.stack(v_x_ego, axis=0)
        h = np.stack(h, axis=0)

        dh = np.gradient(h, dt)

        sa = np.arctan2(dh * self.length, v_x_ego)
        sa[dh == 0.] = 0.

        aa = ActuatorAngle()
        aa.steering_angle = sa
        self.actuator_angle = aa

