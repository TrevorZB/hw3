import csv
import numpy
import multiprocessing
import time


class Regression:
    def __init__(self, filename, step_size):
        self.filename = filename
        self.step_size = step_size
        with open(self.filename, 'r') as data:
            csvreader = csv.reader(data)
            l = len(next(csvreader))
            self.theta = numpy.zeros(l - 1)
            self.theta_trans = self.theta.transpose()
            self.gradient_sum = numpy.zeros(l - 1)

    def _convergence(self):
        for grad in self.gradient_sum:
            if abs(grad) > 0.01:
                return False
        return True

    def main_driver(self):
        i = 0
        s = time.time()
        while True:
            if self._convergence() and i != 0:
                print('Convergence Reached')
                self.print_progress(i)
                break
            self.calc_gradient()
            self.calc_new_theta()
            if i % 10 == 0:
                self.print_progress(i)
            if i % 1000 == 0 and i != 0:
                print(f'TIME: {time.time() - s}')
                break
            i += 1

    def calc_gradient(self):
        with open(self.filename, 'r') as data:
            csvreader = csv.reader(data)
            l = len(next(csvreader))
            self.gradient_sum = numpy.zeros(l - 1)
            for row in csvreader:
                d = numpy.array(row).astype(numpy.float)
                features = d[1:]
                label = d[0]
                self._gradient(label, features)


    def _gradient(self, y_i, x_i):
        frac = None
        ex = -self.theta_trans.dot(x_i)
        try:
            denom = numpy.exp(ex)
        except:
            if ex > 0:
                frac = 0
            else:
                frac = 1
        else:
            frac = 1 / (1 + denom)
        gradient = (y_i - frac) * x_i
        self.gradient_sum += gradient

    def calc_new_theta(self):
        self.theta += (self.step_size * self.gradient_sum)
        self.theta_trans = self.theta.transpose()

    def print_progress(self, i):
        print(f'ITERATION: {i}:')
        # print(f'THETA: {self.theta}')
        print(f'GRADIENT: {self.gradient_sum}')


r = Regression('titanic_data.csv', 0.000001)
r.main_driver()

