#! -*- coding:utf-8 -*-

import sys
import os
import math
import random


class SumTree(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
        self.tree_size = 2**self.tree_level - 1
        self.tree = [0. for _ in range(self.tree_size)]
        self.size = 0
        self.cursor = 0

    def add(self, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.val_update(index, value)

        return index

    def get_val(self, index):
        tree_index = 2**(self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2**(self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        pre_value = value
        if norm:
            value *= self.tree[0]
        list = []
        return self._find(value, 0, pre_value, list)

    def _find(self, value, index, r, list):
        if 2**(self.tree_level - 1) - 1 <= index:
            if index - (2**(self.tree_level - 1) - 1) >= self.size:
                print('!!!!!')
                print(index, value, self.tree[0], r)
                print(list)
                index = (2**(self.tree_level - 1) - 1) + \
                    random.randint(0, self.size)
                #index = (2 ** (self.tree_level - 1) - 1)
            return self.tree[index], index - (2**(self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]
        list.append(left)

        if value <= left + 1e-8:
            return self._find(value, 2 * index + 1, r, list)
        else:
            return self._find(value - left, 2 * (index + 1), r, list)

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2**(k - 1) - 1, 2**k - 1):
                print(self.tree[j])
            print()

    def filled_size(self):
        return self.size
