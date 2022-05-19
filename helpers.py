import math
import numpy as np
Epsilon=0.0000001

def normalize(vector):
    return vector / np.linalg.norm(vector)

class ray:
    def __init__(self, source, direction):
        self.source = source
        self.direction = direction

    def nearest_intersection(self, objects):
        nearest_obj = None
        min_dist = np.inf #infinity
        for obj in objects:
          curr_obj, curr_dist = obj.intersect(self)
          if curr_dist != None:
              if curr_dist < min_dist:
                  min_dist = curr_dist
                  nearest_obj = curr_obj

        return nearest_obj, min_dist

class light:

    def __init__(self, position, light_color,specular_intensity,shadow_intesity,radius):
        self.position = np.array(position)
        self.color = np.array(light_color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intesity
        self.radius = radius

class object:

    def material(self, diffuse, specular, reflection, shininess, tranprancy):
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.shininess = shininess
        self.tranprancy = tranprancy


class plane(object):
    def __init__(self, normal, offset):
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(self, ray):
        N = self.normal
        denominator = np.dot(ray.direction, N)
        # otherwise there is no intersection
        if denominator == 0:
            return None, math.inf
        nominator = -1 * np.dot(ray.source, N) + self.offset
        if nominator<Epsilon and nominator>-Epsilon:
            nominator=0
        t = nominator / denominator
        if t > 0:
            return self, t
        return None, math.inf

    def normalized(self, intersection_point):
        return np.linalg.norm(self.normal)

    def get_normal(self, intersection_point):
        return self.normal

'''
    def intersect(self, ray):
        buttom = np.dot(ray.direction, self.normal)
        if buttom == 0:
            return self, np.inf
        top = -1 * np.dot(ray.source, self.normal) + self.offset
        t = top / buttom
        if t > 0:
            return self, t
        else:
            return None, None

    def normalized(self, intersection_point):
        return np.linalg.norm(self.normal)

    def get_normal(self, intersection_point):
        return self.normal
'''
class sphere(object):
    def __init__(self, center, radius: float):
        self.center = np.array(center)
        self.radius = radius
    def get_normal(self, intersection_point):
        return normalize(intersection_point - self.center)

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)

    def intersect(self, ray):

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, (ray.source - self.center))
        c = np.dot((ray.source - self.center), (ray.source - self.center)) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2 * a)
            t2 = (-b - np.sqrt(delta)) / (2 * a)
            if (t1<Epsilon and t1>-Epsilon) or (t2<Epsilon and t2>-Epsilon):
                return None, None
            if t1 > 0 and t2 > 0:
                return self, min(t1, t2)


        return None, None


class box(object): #cube
    def __init__(self, center, edge_len: float):
        self.center = np.array(center)
        self.edge_len = edge_len

    def intersect(self, ray):

        #TODO

        return None, None

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)

