import numpy as np

class ray:
    def __init__(self, source, direction):
        self.source = source
        self.direction = direction

    def find_closest_object(self, objects):
        nearest_object = None
        min_distance = np.inf

        for obj in objects:
          current_distance, current_object = obj.intersect(self)
          if current_distance != None:
              if current_distance < min_distance:
                  min_distance = current_distance
                  nearest_object = current_object

        return nearest_object, min_distance
