import numpy as np


class MotionEntity(object):
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z


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


# TODO Finish
class Orientation(object):
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Map(object):
    def __init__(self, data=None, homography=None):
        self.data = data
        self.homography = homography

    def to_map_points(self, world_pts):
        org_shape = None
        if len(world_pts.shape) > 2:
            org_shape = world_pts.shape
            world_pts = world_pts.reshape((-1, 2))
        N, dims = world_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = world_pts.T
        map_points = np.fliplr((self.homography @ points_with_one).T[..., :dims])
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points
