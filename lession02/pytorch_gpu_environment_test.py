#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 17:48
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : pytorch_gpu_environment_test.py
# @Software: PyCharm

import torch

if __name__ == "__main__":
    print(torch.__version__)
    print('gpu:',torch.cuda.is_available())