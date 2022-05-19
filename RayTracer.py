import numpy as np
import helpers
import math
import sys
from PIL import Image
import time


lights = []
camera = []
setting = []
material = []
plane = []
spheres = []
box = []
def compute_camera(objects, position, look_at, up_vector, screen_dist, screen_width, img_height, img_width, all_lights):
    screen_height = img_height * screen_width / img_width
    towards = np.array(look_at - position)
    towards_vector = towards / np.linalg.norm(towards)
    p_center = position + (screen_dist * towards_vector)
    background_color = setting[0][0:3]
    screen = (-(img_width/2), img_width/2, -(img_height/2), img_height/2) #left,right,bottom,top
    #the img to return-
    image = np.zeros((img_height, img_width, 3))
    mat = build_matrix(towards_vector[0],towards_vector[1],towards_vector[2])
    up_vector, right_vector = caluclaute_vactors(mat)
    P0 = np.copy(p_center) - screen_width * 0.5 * right_vector - screen_height * 0.5 * up_vector
    right_vector = right_vector * (camera[10] / img_width)
    up_vector = up_vector * (screen_width / img_height)

    #for i, y in enumerate(np.linspace(screen[3], screen[2], img_height)):
    for i in range(img_height):
        P = np.copy(P0)
        for j, x in enumerate(np.linspace(screen[0], screen[1], img_width)):

            print("pixel number", i, j)
            color = np.array(3)
            ray = ray_through_pixel(position, P)
            intersection_point, closest_obj = ray_intersection_object(objects, ray) # Which object the ray hit first
            if closest_obj == None:
                output_color = background_color* np.array([255, 255, 255])
                image[-i, j] = output_color
            else:
                diffuse_and_specular = get_color_light(objects,ray, all_lights, intersection_point, closest_obj, position)
                current_color = (background_color*closest_obj.tranprancy) + (diffuse_and_specular * (1-closest_obj.tranprancy))
                tmp_normal=closest_obj.get_normal(intersection_point)
                reflection = get_reflection(objects,ray,tmp_normal,intersection_point, 1,setting[0][4], background_color)
                #refrection= get_refrection(objects,ray,closest_obj,intersection_point,1,setting[0][4],background_color)
                output_color = current_color+closest_obj.reflection*reflection
                output_color=(np.clip(output_color, 0, 1) * np.array([255, 255, 255]))
                image[-i, j] = output_color
                #print("output_color",output_color)
            P += right_vector
        P0 = P0 + up_vector
    return image


def ray_through_pixel(source, pixel):
    direction = pixel - source
    direction = helpers.normalize(direction)
    ray = helpers.ray(source, direction)
    return ray

def ray_intersection_object(objects, ray):
    closest_obj, min_dist = ray.nearest_intersection(objects)

    if closest_obj is None: # No intersection for this ray
        return np.array((None, None, None)) , None

    intersection_point = ray.source + ((min_dist ) * helpers.normalize(ray.direction))

    return intersection_point, closest_obj

def reflected(vector, normal):
    v = np.array([0, 0, 0])
    normal = helpers.normalize(normal)
    vector = helpers.normalize(vector)
    v = vector - (2 * (np.dot(vector, normal)) * normal)

    return v

def get_color_light(objects, ray, all_lights, intersection_point, closest_obj, camera):

    sum_light = 0

    N = closest_obj.get_normal(intersection_point)  # normal to surface
    for light in all_lights:
        #diffuse
        K_D = np.array(closest_obj.diffuse)
        L=(light.position - intersection_point)
        L = helpers.normalize(L)
        I_L =1#(1-light.shadow_intensity) + light.shadow_intensity*soft_shaddows(objects,L,ray, light.position, light.radius,intersection_point)# fixed for %
        I_D = K_D* light.color * I_L * (np.dot(N, L))
        #spectular
        K_S = np.array(closest_obj.specular)
        V = helpers.normalize(intersection_point-camera)
        n = closest_obj.shininess
        R = reflected((L), N)
        #R2 = L - (np.dot(2*L,N)) * N# reflection of the hit from light to intesection point on the surace
        R = helpers.normalize(R)
        #R2 = helpers.normalize(R2)

        I_S = K_S * (I_L * (np.dot(R, V) ** n))*light.color
        sum_light += (I_D + I_S)
    return sum_light

def get_reflection(objects,ray, normal,intersection_point,level,max_rec,background_color):
    if level < max_rec:
        level += 1
        #print(level)
        reflected_ray_direction = reflected(ray.direction, normal)
        reflected_ray = helpers.ray(intersection_point, reflected_ray_direction)
        new_intersection_reflected_point, new_closest_obj_reflected = ray_intersection_object(objects, reflected_ray)
        if new_closest_obj_reflected != None:
            curr_normal =new_closest_obj_reflected.get_normal(new_intersection_reflected_point)
            diffuse_and_specular = get_color_light(objects, reflected_ray, all_lights, new_intersection_reflected_point, new_closest_obj_reflected,intersection_point)
            return (background_color * new_closest_obj_reflected.tranprancy) + ((1 - new_closest_obj_reflected.tranprancy) * (diffuse_and_specular)) + get_reflection(
            objects, reflected_ray, curr_normal,new_intersection_reflected_point,level,max_rec,background_color) * new_closest_obj_reflected.reflection
        return background_color
    return background_color

def soft_shaddows(objects, L,ray, light_position, light_radius, intersection_point):
    N = setting[0][3]
    rand =np.random.randn(3)
    v1 = rand - np.dot(rand, L) * L
    v1 = v1 / np.linalg.norm(v1)
    if not np.array_equiv(np.dot(v1, L), np.zeros((3, 0, 0))):
        raise Exception()
    v2 = np.cross(L, v1)
    light_source_position = light_position - 0.5 * light_radius * v1 - 0.5 * light_radius * v2
    if not np.allclose(light_source_position + 0.5 * light_radius * v1 + 0.5 * light_radius * v2, light_position):
            raise Exception()
    lights_count = 0
    light_x = v1 * light_radius / N
    light_y = v2 * light_radius / N

    for i in range(int(N)):
        light_source_position_cpy = light_source_position.copy()
        for j in range(int(N)):
            rand_v = light_source_position_cpy + np.random.uniform(0, 1) * light_y
            new_v_direct = intersection_point - rand_v
            new_v_direct = helpers.normalize(new_v_direct)
            new_ray = helpers.ray(rand_v, new_v_direct)
            tmp = new_ray.nearest_intersection(objects)
            closet_obj_arr = [tmp[1], tmp[0], tmp[0].get_normal(intersection_point)]
            #min_primitive_arr = primitives_intersection(random_vector, new_vector_direction)
            if closet_obj_arr[0] < float('inf') and np.allclose(rand_v + closet_obj_arr[0] * new_v_direct, intersection_point, rtol=1e-02, atol=1e-02):
                lights_count += 1
            light_source_position_cpy += light_y
        light_source_position += light_x
    return lights_count / math.pow(N, 2)

def build_matrix(a,b,c):
    Sx = -b
    Cx=math.sqrt(1-(Sx*Sx))
    Sy= -a/Cx
    Cy=c/Cx
    mat = np.array([[Cy, 0, Sy],[-Sx*Sy, Cx, Sx*Cy],[-Cx*Sy, -Sx, Cx*Cy]])
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

def get_lights(): #all lights in scene

    all_lights =[]
    for light in lights:
        our_lights = helpers.light(light[0:3],light[3:6],light[6],light[7],light[8])
        all_lights.append(our_lights)
    return all_lights

def objects(): #all objects in scene
    all_objects = []
    for spher in spheres:
        our_sphere = helpers.sphere(spher[0:3], spher[3])
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

    return all_objects


if __name__ == '__main__':
    start= time.time()
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

    all_objects = objects() # all objects in the scene, by classes defined in helpers
    all_lights = get_lights()
    camera = camera[0]
    image = compute_camera(all_objects, camera[0:3], camera[3:6], camera[6:9], camera[9], camera[10], int(img_h), int(img_w), all_lights)
    im = Image.fromarray(image.astype('uint8'), 'RGB')
    im.save(sys.argv[2])
    im.show()
    print("--- %s seconds ---" % (time.time() - start))