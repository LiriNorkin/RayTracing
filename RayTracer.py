import numpy as np
import helpers
import math
import sys
from PIL import Image



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
    #screen = ((p_center-screen_height/2-screen_width/2), (p_center+screen_height/2+screen_width/2))# left bottom, top right
    #the img to return-
    image = np.zeros((img_height, img_width, 3))
    mat = build_matrix(towards_vector[0],towards_vector[1],towards_vector[2])
    up_vector, right_vector  = caluclaute_vactors(mat)
    #print("mat ", mat)
    #print("right_vector ", right_vector )
    #print("up_vector ", up_vector )
    #print("pcenter", p_center)

    for i, y in enumerate(np.linspace(screen[3], screen[2], img_height)):
        for j, x in enumerate(np.linspace(screen[0], screen[1], img_width)):
            pixel = p_center+x*screen_width/img_width*right_vector+y*screen_height/img_height*up_vector
            color = np.array(3)
            ray = ray_through_pixel(position, pixel) # Shoot a ray through each pixel in the image.
            intersection_point, closest_obj = pixel_color(objects, color, ray, background_color) # Here each pixel color is computed
            light = get_light(ray, all_lights, intersection_point, closest_obj, position)
            #print("light",light)
            final_color = (np.clip(light, 0, 1) * np.array([255, 255, 255]))
            #print("final",final_color)
            output_color = (background_color * closest_obj.tranprancy) + (
                        final_color) * (
                                       1 - closest_obj.tranprancy) + closest_obj.reflection
            image[i, j] = output_color
    return image

    '''
            if(i==499):
                print("i",i,"j",j,"out", output_color,"\n","Pixel",pixel,"V",ray.direction,
                      "intersection_point",intersection_point)
    '''



def ray_through_pixel(source, pixel):
    direction = pixel - source
    direction = helpers.normalize(direction)
    ray = helpers.ray(source, direction)
    return ray

def pixel_color(objects, color, ray, background_color):
    closest_obj, min_dist = ray.nearest_intersection(objects)

    if closest_obj is None: # No intersection for this ray
        return np.array((None, None, None))

    intersection_point = ray.source + ((min_dist - 1e-5) * helpers.normalize(ray.direction))

    # compute color - 1. light 2. material 3. local geomtry

    #output_color = (background_color * closest_obj.tranprancy) + (closest_obj.diffuse + closest_obj.specular) * (1 - closest_obj.tranprancy) + closest_obj.reflection

    #print(output_color)
    return intersection_point, closest_obj # 3 - dim np array\

def reflected(vector, normal):
    v = np.array([0, 0, 0])
    normal = helpers.normalize(normal)
    vector = helpers.normalize(vector)
    v = vector - (2 * (np.dot(vector, normal)) * normal)

    return v

def get_light(ray, all_lights, intersection_point, closest_obj, camera):
    sum_light = 0
    N = closest_obj.get_normal(intersection_point)  # normal to surface
    for light in all_lights:
        #diffuse
        K_D = np.array(closest_obj.diffuse)
        L=(light.position - intersection_point)
        L = helpers.normalize(L)
        #print("lightposition",light.position)
        #print("inter",intersection_point)
        I_L = (1-light.shadow_intensity) + light.shadow_intensity*(1) # fixed for %
        I_D = light.color *K_D * I_L * (np.dot(N, L))
        #print("L",L)
        #print("dot", np.dot(N, L))
        #spectular
        K_S = np.array(closest_obj.specular)
        V = helpers.normalize(camera-intersection_point)
        #print("V",V)
        n = closest_obj.shininess
        #print("n",n)
        R = reflected(((-1) * L), N)  # reflection of the hit from light to intesection point on the surace
        R = helpers.normalize(R)
        I_S = (K_S * I_L * (np.dot(V, R) ** closest_obj.shininess))
        #print("IS", I_S)
        sum_light += (I_D + I_S) * light.color

    #print("Sum=",sum_light)
    return sum_light


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

    scene_file, output_image_file, img_w, img_h = get_args()
    scene_file = open(scene_file, "r")
    lines = scene_file.readlines()
    for line in lines:
        if line[0] == '#' or line[0] == '\n':
            continue
        else:
            line = line.strip().split("\t")
            #line = [x.split() for x in line.split("\t") if x != ""]
            #print(line)
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
    im.save("your_file.jpeg")