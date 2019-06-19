import pygame as pg
import matrix
from pygame import gfxdraw
import sys


class Graphics:
    def __init__(self, i, h, o):
        self.__num_i_nodes = i
        self.__num_h_nodes = h
        self.__num_o_nodes = o

        self.__SCREEN_WIDTH = 640
        self.__SCREEN_HEIGHT = 480
        self.white = (255, 255, 255)
        self.black = (75, 75, 75)
        self.red = (255, 0, 0)

        self.__i_node_spacing = self.__SCREEN_HEIGHT / (i+1)
        self.__h_node_spacing = self.__SCREEN_HEIGHT / (h+1)
        self.__o_node_spacing = self.__SCREEN_HEIGHT / (o+1)

        self.__i_nodes = []
        self.__h_nodes = []
        self.__o_nodes = []
        self.__ih_weights = []
        self.__ho_weights = []

        self.init_i_nodes()
        self.init_h_nodes()
        self.init_o_nodes()
        self.init_ih_weights()
        self.init_ho_weights()

    # initialize

    def init_i_nodes(self):
        y = self.__i_node_spacing
        for i in range(self.__num_i_nodes):
            self.__i_nodes.append(Node(self.__SCREEN_WIDTH/8, y, 10))
            y += self.__i_node_spacing

    def init_h_nodes(self):
        y = self.__h_node_spacing
        for i in range(self.__num_h_nodes):
            self.__h_nodes.append(Node(self.__SCREEN_WIDTH/2, y, 10))
            y += self.__h_node_spacing

    def init_o_nodes(self):
        y = self.__o_node_spacing
        for i in range(self.__num_o_nodes):
            self.__o_nodes.append(Node(self.__SCREEN_WIDTH/8 * 7, y, 10))
            y += self.__o_node_spacing

    def init_ih_weights(self):
        for i in range(len(self.__i_nodes)):
            for j in range(len(self.__h_nodes)):
                self.__ih_weights.append(Weight((self.__i_nodes[i].x, self.__i_nodes[i].y), (self.__h_nodes[j].x, self.__h_nodes[j].y)))

    def init_ho_weights(self):
        for i in range(len(self.__h_nodes)):
            for j in range(len(self.__o_nodes)):
                self.__ho_weights.append(Weight((self.__h_nodes[i].x, self.__h_nodes[i].y), (self.__o_nodes[j].x, self.__o_nodes[j].y)))

    # draw

    def draw_input_nodes(self, screen, black, red):
        for i in range(len(self.__i_nodes)):
            if self.__i_nodes[i].data >= 0: color = black
            else: color = red
            pg.gfxdraw.filled_circle(screen, int(self.__i_nodes[i].x), int(self.__i_nodes[i].y), self.__i_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__i_nodes[i].x), int(self.__i_nodes[i].y), self.__i_nodes[i].size, color)

    def draw_hidden_nodes(self, screen, black, red):
        for i in range(len(self.__h_nodes)):
            if self.__h_nodes[i].data >= 0: color = black
            else: color = red
            pg.gfxdraw.filled_circle(screen, int(self.__h_nodes[i].x), int(self.__h_nodes[i].y), self.__h_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__h_nodes[i].x), int(self.__h_nodes[i].y), self.__h_nodes[i].size, color)

    def draw_output_nodes(self, screen, black, red):
        for i in range(len(self.__o_nodes)):
            if self.__o_nodes[i].data >= 0: color = black
            else: color = red
            pg.gfxdraw.filled_circle(screen, int(self.__o_nodes[i].x), int(self.__o_nodes[i].y), self.__o_nodes[i].size, color)
            pg.gfxdraw.aacircle(screen, int(self.__o_nodes[i].x), int(self.__o_nodes[i].y), self.__o_nodes[i].size, color)

    def draw_ih_weights(self, screen, black, red):
        for i in range(len(self.__ih_weights)):
            if self.__ih_weights[i].data >= 0: color = black
            else: color = red
            pg.draw.aaline(screen, color, self.__ih_weights[i].coor1, self.__ih_weights[i].coor2)

    def draw_ho_weights(self, screen, black, red):
        for i in range(len(self.__ho_weights)):
            if self.__ih_weights[i].data >= 0: color = black
            else: color = red
            pg.draw.aaline(screen, color, self.__ho_weights[i].coor1, self.__ho_weights[i].coor2)

    # update
            
    def update_input_nodes(self, arr):
        for i in range(len(arr)):
            self.__i_nodes[i].data = arr[i]

    def update_hidden_nodes(self, arr):
        for i in range(len(arr)):
            self.__h_nodes[i].data = arr[i]

    def update_output_nodes(self, arr):
        for i in range(len(arr)):
            self.__o_nodes[i].data = arr[i]

    def update_ih_weights(self, arr):
        for i in range(len(arr)):
            self.__ih_weights[i].data = arr[i]

    def update_ho_weights(self, arr):
        for i in range(len(arr)):
            self.__ho_weights[i].data = arr[i]

    # draw screen

    def draw(self, input, hidden, output, __weights_ih, __weights_ho):
        pg.init()
        size = (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT)
        screen = pg.display.set_mode(size)
        pg.display.set_caption("Neural Network Visualization")

        game_exit = False
        while not game_exit:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit(0)

            screen.fill(self.white)
            self.update_input_nodes(matrix.Matrix.to_array(input))
            self.update_hidden_nodes(matrix.Matrix.to_array(hidden))
            self.update_output_nodes(matrix.Matrix.to_array(output))
            self.update_ih_weights(matrix.Matrix.to_array(__weights_ih))
            self.update_ho_weights(matrix.Matrix.to_array(__weights_ho))
            self.draw_ih_weights(screen, self.black, self.red)
            self.draw_ho_weights(screen, self.black, self.red)
            self.draw_input_nodes(screen, self.black, self.red)
            self.draw_hidden_nodes(screen, self.black, self.red)
            self.draw_output_nodes(screen, self.black, self.red)
            pg.display.update()
            game_exit = True


class Node:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.data = 0.0


class Weight:
    def __init__(self, coor1, coor2):
        self.coor1 = coor1
        self.coor2 = coor2
        self.data = 0.0