#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 18:04
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : gradient_descent.py
# @Software: PyCharm

import numpy as np


# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    """
    计算均方误差
    :param b:
    :param w:
    :param points:
    :return:
    """
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    """
    计算梯度
    :param b_current:
    :param w_current:
    :param points:
    :param learningRate:
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -2/N * (y - (w_current * x) + b_current)
        w_gradient += -2/N * x * (y - (w_current * x) + b_current)
    b_new = b_current - (learningRate * b_gradient)
    w_new = w_current - (learningRate * w_gradient)
    return [b_new, w_new]


def gradient_descent_runner(points, b_starting, w_starting, learningRate, num_interations):
    """
    运行梯度迭代计算
    :param points:
    :param b_starting:
    :param w_starting:
    :param learningRate:
    :param num_interations:
    :return:
    """
    b = b_starting
    w = w_starting
    for i in range(num_interations):
        b, w = step_gradient(b, w, np.array(points), learningRate)
    return [b, w]


if __name__ == "__main__":
    points = np.genfromtxt("data.csv", delimiter=",")
    learningRate = 0.0001
    b_init = 0  # 初始化y截距
    w_init = 0  # 初始化斜率
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(b_init, w_init,
                  compute_error_for_line_given_points(b_init, w_init, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, b_init, w_init, learningRate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )
