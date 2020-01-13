import numpy as np
from scipy.ndimage.interpolation import rotate


class MotionEntity(object):
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z
        self.m = None

    @property
    def l(self):
        if self.z is not None:
            return np.linalg.norm(np.vstack((self.x, self.y, self.z)), axis=0)
        else:
            return np.linalg.norm(np.vstack((self.x, self.y)), axis=0)


class Position(MotionEntity):
    pass


class Velocity(MotionEntity):
    @staticmethod
    def from_position(position, dt=1):
        dx = np.zeros_like(position.x) * np.nan
        dx[~np.isnan(position.x)] = np.gradient(position.x[~np.isnan(position.x)], dt)

        dy = np.zeros_like(position.y) * np.nan
        dy[~np.isnan(position.y)] = np.gradient(position.y[~np.isnan(position.y)], dt)

        if position.z is not None:
            dz = np.zeros_like(position.z) * np.nan
            dz[~np.isnan(position.z)] = np.gradient(position.z[~np.isnan(position.z)], dt)
        else:
            dz = None

        return Velocity(dx, dy, dz)


class Acceleration(MotionEntity):
    @staticmethod
    def from_velocity(velocity, dt=1):
        ddx = np.zeros_like(velocity.x) * np.nan
        ddx[~np.isnan(velocity.x)] = np.gradient(velocity.x[~np.isnan(velocity.x)], dt)

        ddy = np.zeros_like(velocity.y) * np.nan
        ddy[~np.isnan(velocity.y)] = np.gradient(velocity.y[~np.isnan(velocity.y)], dt)

        if velocity.z is not None:
            ddz = np.zeros_like(velocity.z) * np.nan
            ddz[~np.isnan(velocity.z)] = np.gradient(velocity.z[~np.isnan(velocity.z)], dt)
        else:
            ddz = None

        return Acceleration(ddx, ddy, ddz)


class ActuatorAngle(object):
    def __init__(self):
        pass


class Scalar(object):
    def __init__(self, value):
        self.value = value
        self.derivative = None

# TODO Finish
class Orientation(object):
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Map(object):
    def __init__(self, data=None, homography=None, description=None, data_file=""):
        self.data = data
        self.homography = homography
        self.description = description
        self.uint = False
        self.data_file = data_file
        self.rotated_maps_origin = None
        self.rotated_maps = None
        if self.data.dtype == np.uint8:
            self.uint = True

    @property
    def fdata(self):
        if self.uint:
            return self.data / 255.
        else:
            return self.data

    def to_map_points(self, world_pts):
        org_shape = None
        if len(world_pts.shape) > 2:
            org_shape = world_pts.shape
            world_pts = world_pts.reshape((-1, 2))
        N, dims = world_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = world_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims] # TODO There was np.fliplr here for pedestrian dataset. WHY?
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points

    def to_rotated_map_points(self, world_pts, rotation_angle):
        rotation_rad = -rotation_angle * np.pi / 180
        rot_mat = np.array([[np.cos(rotation_rad), np.sin(rotation_rad), 0.],
                            [-np.sin(rotation_rad), np.cos(rotation_rad), 0.],
                            [0., 0., 1.]])
        org_map_points = self.to_map_points(world_pts) + 1

        org_shape = None
        if len(org_map_points.shape) > 2:
            org_shape = org_map_points.shape
            org_map_points = org_map_points.reshape((-1, 2))
        N, dims = org_map_points.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = org_map_points.T
        org_map_pts_rot = (rot_mat @ points_with_one).T[..., :dims]
        if org_shape is not None:
            org_map_pts_rot = org_map_pts_rot.reshape(org_shape)

        map_pts_rot = self.rotated_maps_origin + org_map_pts_rot
        return map_pts_rot

    def calculate_rotations(self):
        org_shape = self.data.shape
        l = (np.ceil(np.sqrt(org_shape[0]**2 + org_shape[1]**2)) * 2).astype(int) + 1
        rotated_maps = np.zeros((360, l, l, org_shape[2]), dtype=np.uint8)
        o = np.array([l // 2, l // 2])
        rotated_maps[0, o[0]+1:o[0]+org_shape[0]+1, o[1]+1:o[1]+org_shape[1]+1] = self.data
        for i in range(1, 360):
            rotated_maps[i] = rotate(rotated_maps[0], reshape=False, angle=i, prefilter=False)
        rotated_maps[0] = rotate(rotated_maps[0], reshape=False, angle=0, prefilter=False)
        self.rotated_maps_origin = o
        self.rotated_maps = rotated_maps

    # def __getstate__(self):
    #     with open(self.data_file, 'w') as f:
    #         np.save(f, self.rotated_maps)
    #     self.rotated_maps = None
    #     state = self.__dict__.copy()
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     with open(self.data_file, 'r') as f:
    #         self.rotated_maps = np.load(f)

if __name__ == "__main__":
    img = np.zeros((103, 107, 3))
    img[57, 84] = 255.
    homography = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    m = Map(data=img, homography=homography)
    m.calculate_rotations()
    t = m.to_rotated_map_points(np.array([[57, 84]]), 0).astype(int)
    print(m.rotated_maps[0, t[0, 0], t[0, 1]])