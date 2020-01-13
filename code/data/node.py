import numpy as np

class Node(object):
    def __init__(self, type, position=None, velocity=None, acceleration=None, orientation=None, first_timestep=0):
        self.type = type
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.acceleration = acceleration
        self.first_timestep = first_timestep
        self.dimensions = ['x', 'y', 'z']
        self.is_robot = False
        self._last_timestep = None

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

    def get(self, ts_scene, entities, dims, padding=np.nan):
        if isinstance(entities, list):
            return np.hstack([self.get_entity(ts_scene, entity, dims, padding) for entity in entities])
        else:
            return self.get_entity(ts_scene, entities, dims, padding)

    @property
    def timesteps(self):
        return self.position.x.size  # TODO: Check all for same size?

    @property
    def last_timestep(self):
        if self._last_timestep is None:
            self._last_timestep = self.first_timestep + self.timesteps - 1
        return self._last_timestep
