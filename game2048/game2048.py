# coding: utf-8

import argparse
import pyglet
from pyglet.window import key
import random
import copy

class Game2048(object):
    def reset(self):
        self.board_number = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        self.pre_board_number = copy.deepcopy(self.board_number)

        self.score = 0
        self.pre_score = 0

        self.finish_flag = False
        self.enter = False
        self.pressed = False

    def __init__(self, render=True):
        self.render = render

        self.window_width = 540
        self.window_height = 640

        self.board_size = 500
        self.board_x_offset = (self.window_width - self.board_size) / 2.0
        self.board_y_offset = 0

        self.reset()

        self.board_color = [[197, 185, 173], [229 ,219, 209], [228, 215, 192], [233, 170, 116], [235, 143, 95], [236, 119, 91], [236, 90, 57], [228, 199, 110], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]

        if self.render:
            self.window = pyglet.window.Window(self.window_width, self.window_height)
            self.window.set_caption('This is a pyglet sample')
        else:
            self.window = pyglet.window.Window(1, 1)

        self.quad_list = []
        self.number_list = []
        for i in range(len(self.board_number)):
            self.quad_list.append([])
            self.number_list.append([])
            for j in range(len(self.board_number[i])):

                d = 5 * 1 + 4 * 10
                x = self.board_x_offset + self.board_size / d * 1 + self.board_size / d * i * (10 + 1)
                y = self.board_y_offset + self.board_size / d * 1 + self.board_size / d * (4 - j) * (10 + 1)
                box_size = self.board_size / d * 10

                board_number = "" if self.board_number[j][i] == 0 else str(2**self.board_number[j][i])
                self.number_list[i].append(pyglet.text.Label(board_number, font_name="Arial", font_size=32, x=x + box_size / 2.0, y=y + box_size / 2.0, anchor_x='center', anchor_y='center', color=(0,0,0,255))) #, batch=batch,group=None))
                self.quad_list[i].append(pyglet.graphics.vertex_list(4,
                                                                ('v2f', [x, y, x + box_size, y, x + box_size, y + box_size, x, y + box_size]),
                                                                ('c3B', self.board_color[0] * 4)))

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.UP:
                if not self.pressed:
                    self.step(0)
                self.pressed = True
            elif symbol == key.RIGHT:
                if not self.pressed:
                    self.step(1)
                self.pressed = True
            elif symbol == key.DOWN:
                if not self.pressed:
                    self.step(2)
                self.pressed = True
            elif symbol == key.LEFT:
                if not self.pressed:
                    self.step(3)
                self.pressed = True

            if symbol == key.RETURN:
                self.enter = True

        @self.window.event
        def on_key_release(symbol, modifiers):
            self.enter = False
            self.pressed = False

        '''
        @self.window.event
        def on_draw():
            self.window.clear()
            for i in range(len(self.board_number)):
                for j in range(len(self.board_number[i])):
                    color = [255, 255, 255]
                    self.quad_list[i][j].colors = color * 4
                    self.quad_list[i][j].draw(pyglet.gl.GL_QUADS)

                    board_number = "" if self.board_number[j][i] == 0 else str(2**self.board_number[j][i])
                    self.number_list[i][j].text = board_number
                    self.number_list[i][j].draw()
        '''

    def calc_empty(self, board_number):
        sum_empty = 0
        for i in range(4):
            for j in range(4):
                if board_number[i][j] == 0:
                    sum_empty += 1
        return sum_empty

    def calc_max(self, board_number):
        max_num = 0
        for i in range(4):
            for j in range(4):
                if board_number[i][j] > max_num:
                    max_num = board_number[i][j]
        return max_num

    def make_window(self):
        self.render = True
        self.window.width = self.window_width
        self.window.height = self.window_height
        self.window.set_caption('This is a pyglet sample')

    def draw(self):
        if not self.render:
            return

        self.window.clear()
        #self.window.switch_to()
        self.window.dispatch_events()

        for i in range(len(self.board_number)):
            for j in range(len(self.board_number[i])):
                color = self.board_color[self.board_number[j][i]]
                self.quad_list[i][j].colors = color * 4
                self.quad_list[i][j].draw(pyglet.gl.GL_QUADS)

                board_number = "" if self.board_number[j][i] == 0 else str(2**self.board_number[j][i])
                self.number_list[i][j].text = board_number
                self.number_list[i][j].draw()

        self.window.flip()

    def update_line(self, line):
        #print("start")
        #print(line)
        score = 0
        res_line = []
        i = 0
        while i < 4:
            if line[i] == 0:
                i += 1
                continue
            if i == 3:
                res_line.append(line[i])
                break
            for j in range(i + 1, 4):
                if line[j] == 0:
                    if j == 3:
                        res_line.append(line[i])
                        i += 1
                    continue
                if line[i] == line[j]:
                    n = line[i] + 1
                    score += 2**n
                    res_line.append(n)
                    i = j + 1
                    break
                else:
                    res_line.append(line[i])
                    i = j
                    break
        res_line += [0 for i in range(4 - len(res_line))]
        #print(res_line)
        #print("end")
        return res_line, score

    def update_board(self, direction, update=True):
        self.pre_board_number = copy.deepcopy(self.board_number)
        self.pre_score = self.score
        score = 0

        if direction == 0:   # up
            invalid_flag = True
            for i in range(4):
                pre_line = [b[i] for b in self.board_number]
                pro_line, tmp_score = self.update_line(pre_line)
                score += tmp_score
                if pre_line != pro_line:
                    invalid_flag = False
                if update:
                    for j in range(4):
                        self.board_number[j][i] = pro_line[j]
                        self.score += score
            if invalid_flag:
                return -1
        elif direction == 1:   # right
            invalid_flag = True
            for i in range(4):
                pre_line = [b for b in self.board_number[i]]
                pre_line.reverse()
                pro_line, tmp_score = self.update_line(pre_line)
                score += tmp_score
                if pre_line != pro_line:
                    invalid_flag = False
                pro_line.reverse()
                if update:
                    for j in range(4):
                        self.board_number[i][j] = pro_line[j]
                        self.score += score
            if invalid_flag:
                return -1
        elif direction == 2:   # down
            invalid_flag = True
            for i in range(4):
                pre_line = [b[i] for b in self.board_number]
                pre_line.reverse()
                pro_line, tmp_score = self.update_line(pre_line)
                score += tmp_score
                if pre_line != pro_line:
                    invalid_flag = False
                pro_line.reverse()
                if update:
                    for j in range(4):
                        self.board_number[j][i] = pro_line[j]
                        self.score += score
            if invalid_flag:
                return -1
        elif direction == 3:   # left
            invalid_flag = True
            for i in range(4):
                pre_line = [b for b in self.board_number[i]]
                pro_line, tmp_score = self.update_line(pre_line)
                score += tmp_score
                if pre_line != pro_line:
                    invalid_flag = False
                if update:
                    for j in range(4):
                        self.board_number[i][j] = pro_line[j]
                        self.score += score
            if invalid_flag:
                return -1
        return 0

    def step(self, direction, pop_number=True):
        invalid_flag = self.update_board(direction, update=True)
        if invalid_flag == -1:
            return -1

        #print("score : {}".format(self.score))
        if pop_number:
            # check if game is finished
            finish_flag = True
            for i in range(4):
                for j in range(4):
                    if self.board_number[i][j] == 0:
                        finish_flag = False
                        break
                if not finish_flag:
                    break
            if finish_flag:
                print("finish by no area remaining")
                self.finish_flag = True
                #print("score : {}, max_number : {}".format(self.score, 2**max([max(ns) for ns in self.board_number])))
                return 0

            while True:
                x = random.randint(0, 3)
                y = random.randint(0, 3)
                if self.board_number[x][y] == 0:
                    n = random.randint(1, 2)
                    self.board_number[x][y] = n
                    break

            # check if game is finished again
            finish_flag = True
            for i in range(4):
                for j in range(4):
                    if self.board_number[i][j] == 0:
                        finish_flag = False
                        break
                if not finish_flag:
                    break
            if finish_flag:
                for i in range(4):
                    if self.update_board(i, update=False) == 0:
                        #print("safe with directoin: {}".format(i))
                        finish_flag = False
                        break
            if finish_flag:
                print("finish by not be able to move")
                self.finish_flag = True
                #print("score : {}, max_number : {}".format(self.score, 2**max([max(ns) for ns in self.board_number])))
                return 0

    def loop(self, manual=False):
        while not self.finish_flag:
            self.draw()
            if not manual:
                a = random.randint(0, 3)
                self.step(a)
        #print("score : {}".format(self.score))
        self.close()

    def close(self):
        if not self.render:
            return

        print("press enter")
        while not self.enter:
            self.draw()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--automatic', action='store_true', help='automatic random mode (default)')
    parser.add_argument('-m', '--manual', action='store_true', help='manual mode')
    args = parser.parse_args()

    game = Game2048(True)

    if args.automatic:
        game.loop(False)
    elif args.manual:
        game.loop(True)
    else:
        game.loop(False)

if __name__ == "__main__":
    main()
