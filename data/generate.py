import sys
import random
from math import sqrt

WIDTH = 100
num_spheres = 1
output = "output.txt"

def getDistanceToPoint(x1,y1,z1,x2,y2,z2):
    deltx = x2-x1
    delty = y2-y1
    deltz = z2-z1
    return sqrt((deltx ** 2) + (delty ** 2) + (deltz ** 2))

if __name__ == "__main__":
    # pass in num spheres
    if (len(sys.argv) == 2):
        num_spheres = int(sys.argv[1])

    if (num_spheres <= 0):
        print("num spheres must be a positive integer")
        sys.exit(1)

    print(f"Number of spheres: {num_spheres}")

    # zero initialized 3D grid of width 1000
    grid = [[[0] * WIDTH] * WIDTH] * WIDTH

    # generate tuples of randomized sphere centers and radii
    # ( (x,y,z), radius )
    sphere_info = []
    for _ in range(num_spheres):
        x = random.randint(0,WIDTH)
        y = random.randint(0,WIDTH)
        z = random.randint(0,WIDTH)
        radius = random.randint(1, WIDTH/2)
        sphere_info.append(((x, y, z), radius))


    print("generated sphere info:")
    for ((x,y,z), radius) in sphere_info:
        print(f"center: (x={x},y={y},z={z}), radius: {radius}")

    # for every point in the grid, if the point's distance to the center of a
    # sphere is shorter than the sphere's radius, set the point to 1
    for x1 in range(WIDTH):
        for y1 in range(WIDTH):
            for z1 in range(WIDTH):
                for ((x2,y2,z2), radius) in sphere_info:
                    dToCenter = getDistanceToPoint(x1,y1,z1,x2,y2,z2)
                    if dToCenter < radius:
                        grid[x1][y1][z1] = 1


    print(f"writing to file: {output}")
    # write grid to file
    with open(output, 'w') as outFile:
        # first line is the width of grid
        outFile.write(f"{WIDTH}\n")

        # write points in format x, y, z, value
        for x in range(WIDTH):
            for y in range(WIDTH):
                for z in range(WIDTH):
                    outFile.write(f"{x} {y} {z} {grid[x][y][z]}\n")
