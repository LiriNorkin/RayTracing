import numpy as np

class ray:
    def __init__(self, source, direction):
        self.source = source
        self.direction = direction

    def nearest_intersection(self, objects):
        nearest_object = None
        min_dist = np.inf

        for obj in objects:
          current_distance, current_object = obj.intersect(self)
          if current_distance != None:
              if current_distance < min_distance:
                  min_distance = current_distance
                  nearest_object = current_object

        return nearest_object, min_distance


class object:

    def material(self, diffuse, specular, reflection, shininess, tranprancy):
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.shininess = shininess
        self.tranprancy = tranprancy


class plane(object):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None, None

    def normalized(self, intersection_point):
        return np.linalg.norm(self.normal)

class sphere(object):
    def __init__(self, center, radius: float):
        self.center = np.array(center)
        self.radius = radius


    def intersect(self, ray):

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, (ray.origin - self.center))
        c = np.dot((ray.origin - self.center), (ray.origin - self.center)) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2 * a)
            t2 = (-b - np.sqrt(delta)) / (2 * a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2), self

        return None, None

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)


class box(object):
    def __init__(self, center, edge_len: float):
        self.center = np.array(center)
        self.edge_len = edge_len

    def intersect(self, ray):

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, (ray.origin - self.center))
        c = np.dot((ray.origin - self.center), (ray.origin - self.center)) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2 * a)
            t2 = (-b - np.sqrt(delta)) / (2 * a)
            if t1 > 0 and t2 > 0:
                return min(t1, t2), self

        return None, None

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)

