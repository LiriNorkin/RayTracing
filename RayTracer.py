import numpy as np
import helpers


def compute_camera(position, look_at, up_vector, screen_dist, screen_width, img_height = 500, img_width = 500):
    screen_height = img_height * screen_width / img_width
    towards = look_at - position
    p_center = position + screen_dist * towards

    screen = ((p_center-screen_height/2-screen_width/2), (p_center+screen_height/2+screen_width/2))# left bottom, top right
    #the img to return
    image = np.zeros((img_height, img_width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], img_height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], img_width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)

            # Here each pixel color is computed
            ray = ray_through_pixel(position, pixel)
            color = pixel_color()

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def ray_through_pixel(source, pixel):
    direction = pixel - source
    direction = direction / np.linalg.norm(direction) # norm
    curr_ray = helpers.ray(source, direction)
    return curr_ray

def pixel_color():
    return