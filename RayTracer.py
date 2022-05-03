import numpy as np
import helpers
import  math
import sys

camera = []
setting = []
material = []
plane = []
spheres = []
box = []
lights = []


def compute_camera(position, look_at, up_vector, screen_dist, screen_width, img_height = 500, img_width = 500):
    screen_height = img_height * screen_width / img_width
    towards = np.array(look_at - position)
    p_center = position + (screen_dist * towards)
    towards_vector = towards / np.linalg.norm(towards)
    screen = (-(img_width/2), img_width/2, -(img_height/2), img_height/2) #left,right,bottom,top
    #screen = ((p_center-screen_height/2-screen_width/2), (p_center+screen_height/2+screen_width/2))# left bottom, top right
    #the img to return
    image = np.zeros((img_height, img_width, 3))
    mat = build_matrix(towards_vector[0],towards_vector[1],towards_vector[2])
    up_vector, right_vector  = caluclaute_vactors(mat)
    for i, y in enumerate(np.linspace(screen[3], screen[2], img_height)):
        for j, x in enumerate(np.linspace(screen[0], screen[1], img_width)):
            pixel = p_center+i*screen_width/img_width*right_vector+j*img_height
            color = np.zeros(3)

            # Here each pixel color is computed
            ray = ray_through_pixel(position, pixel)
            color = pixel_color()

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def ray_through_pixel(source, pixel):
    direction = pixel - source
   # direction = direction / np.linalg.norm(direction) # norm
    curr_ray = helpers.ray(source, direction)
    return curr_ray

def pixel_color():
    return

def build_matrix(a,b,c):
    Sx = -b
    Cx=math.sqrt(1-math.sqrt(Sx))
    Sy= -a/Cx
    Cy=c/Cx
    mat = np.array[[Cy, 0, Sy][-Sx*Sy, Cx, Sx*Cy][-Cx*Sy, -Sx, Cx*Cy]]
    return mat

def caluclaute_vactors(matrix):
    up_vector = np.array([0,1,0])
    up_vector = up_vector.dot(matrix)
    right_vector = np.array([1,0,0])
    right_vector = right_vector.dot(matrix)
    return up_vector, right_vector



def get_args():
    """A Helper function that defines the program arguments."""
    args = sys.argv[1:]
    img_h = 500
    img_w = 500
    scene_file = args[0]
    output_image_file = args[1]

    if len(args) > 2:
        img_w = args[2]
    if len(args) > 3:
        img_h = args[3]

    return scene_file, output_image_file, img_w, img_h

def objects(): #all objects in scene
    all_objects = []
    for spher in spheres:
        our_sphere = helpers.sphere(spher[0:3], spher[3] )
        mat = material[int(spher[4]) - 1]
        our_sphere.material(mat[0:3], mat[3:6], mat[6:9], mat[9], mat[10])
        all_objects.append(our_sphere)

    for pln in plane:
        our_pln = helpers.plane(pln[0:3], pln[3])
        mat = material[int(pln[4])-1]
        our_pln.material(mat[0:3], mat[3:6], mat[6:9], mat[9], mat[10])
        all_objects.append(our_pln)

    for bx in box:
        our_box = helpers.box(bx[0:3], bx[3])
        mat = material[int(pln[4]) - 1]
        our_box.material(mat[0:3], mat[3:6], mat[6:9], mat[9], mat[10])
        all_objects.append(our_box)

    print(all_objects)
    return all_objects

if __name__ == '__main__':
    objects()

    scene_file, output_image_file, img_w, img_h = get_args()
    scene_file = open(scene_file, "r")
    lines = scene_file.readlines()
    for line in lines:
        if line[0] == '#' or line[0] == '\n':
            continue
        else:
            line = line.strip().split("\t")
            title = line[0]

            for word in line:
                word=word.strip().replace(" ","").strip("\n")
                if word != '' and word != title and word!='cam' and word!= 'set':
                    word = float(word)
                    if title == 'cam ':
                        camera.append(word)
                    elif title == 'set ':
                        setting.append(word)
                    elif title == 'mtl':
                        material.append(word)
                    elif title == 'pln':
                        plane.append(word)
                    elif title == 'sph':
                        spheres.append(word)
                    elif title == 'lgt':
                        lights.append(word)
                    elif title == 'box':
                        box.append(word)

    camera = np.array(camera, dtype=float).reshape((int(len(camera)/11), 11))
    setting = np.array(setting, dtype=float).reshape((int(len(setting)/5), 5))
    material = np.array(material, dtype=float).reshape((int(len(material)/11), 11))
    plane = np.array(plane, dtype=float).reshape((int(len(plane)/5), 5))
    spheres = np.array(spheres, dtype=float).reshape((int(len(spheres)/5), 5))
    box = np.array(box, dtype=float).reshape((int(len(box)/5), 5))
    lights = np.array(lights, dtype=float).reshape((int(len(lights)/9), 9))
    objects()

def find_intersection(ray, objects):
    nearest_object, min_distance = ray.nearest_intersection(ray, objects)

    if nearest_object is None:
        return None, None, None

    intersection_point = ray.origin + ((min_distance-1e-5) * np.linalg.norm(ray.direction))

    return intersection_point, nearest_object, min_distance
