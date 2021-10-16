import numpy as np
import random
import components.sum_tree as sum_tree


class Experience(object):
    """ The class represents prioritized experience replay buffer.

        The class has functions: store samples, pick samples with
        probability in proportion to sample's priority, update
        each sample's priority, reset alpha.

        see https://arxiv.org/pdf/1511.05952.pdf .

        """

    def __init__(self, memory_size, alpha=1):
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha

    def add(self, priority):
        index = self.tree.add(priority**self.alpha)
        return index

    def select(self, batch_size):

        if self.tree.filled_size() < batch_size:
            return None

        indices = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            priority, index = self.tree.find(r)
            priorities.append(priority)
            indices.append(index)
            self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities

        return indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

                Parameters
                ----------
                indices :
                        list of sample indices
                """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
