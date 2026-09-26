"""CEC2017 two-task benchmarks. External benchmark data are required."""
from pathlib import Path
import scipy.io as sio
import numpy as np
import math

def _prepare_rotated_shifted(array, M, opt):
    x = np.asarray(array, dtype=np.float64).reshape(-1)
    opt = np.asarray(opt, dtype=np.float64).reshape(-1)
    M = np.asarray(M, dtype=np.float64)
    if x.size != opt.size:
        raise ValueError(f'x size {x.size} != opt size {opt.size}')
    if M.shape != (x.size, x.size):
        raise ValueError(f'M shape {M.shape} != ({x.size}, {x.size})')
    x = x - opt
    x = (M @ x).reshape(-1)
    return x

def Sphere(array, opt):
    array = array - opt
    return np.sum(array ** 2)

def Rosenbrock(array):
    return np.sum(100 * (array[1:] - array[:-1] ** 2) ** 2 + (1 - array[:-1]) ** 2)

def Ackley(array, M, opt):
    x = _prepare_rotated_shifted(array, M, opt)
    res = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
    res = res - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20.0 + np.exp(1.0)
    return res

def Rastrigin(array, M, opt):
    x = _prepare_rotated_shifted(array, M, opt)
    d = x.size
    return 10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))

def Griewank(array, M, opt):
    x = _prepare_rotated_shifted(array, M, opt)
    d = x.size
    i = np.arange(1, d + 1, dtype=np.float64)
    return 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i)))

def Weierstrass(array, M, opt):
    x = _prepare_rotated_shifted(array, M, opt)
    d = x.size
    a = 0.5
    b = 3.0
    kmax = 20
    result = 0.0
    for i in range(d):
        for k in range(kmax + 1):
            result += a ** k * math.cos(2 * math.pi * b ** k * (x[i] + 0.5))
    c = 0.0
    for k in range(kmax + 1):
        c += a ** k * math.cos(2 * math.pi * b ** k * 0.5)
    return result - d * c

def Schwefel(array):
    d = len(array)
    return 418.9829 * d - np.sum(array * np.sin(np.sqrt(np.abs(array))))

def CIHS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-100, 100, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Griewank(x, M, opt)
    elif task == 2:
        M = data['Rotation_Task2']
        opt = data['GO_Task2']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rastrigin(x, M, opt)

def CIMS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        M = data['Rotation_Task2']
        opt = data['GO_Task2']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rastrigin(x, M, opt)

def CILS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        l, u, d = (-500, 500, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Schwefel(x)

def PIHS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rastrigin(x, M, opt)
    elif task == 2:
        opt = data['GO_Task2']
        l, u, d = (-100, 100, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Sphere(x, opt)

def PIMS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rosenbrock(x)

def PILS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        M = data['Rotation_Task2']
        opt = data['GO_Task2']
        l, u, d = (-0.5, 0.5, 25)
        x = l + array * (u - l)
        x = x[:d]
        return Weierstrass(x, M, opt)

def NIHS(array, task, data):
    if task == 1:
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rosenbrock(x)
    elif task == 2:
        M = data['Rotation_Task2']
        opt = data['GO_Task2']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rastrigin(x, M, opt)

def NIMS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-100, 100, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Griewank(x, M, opt)
    elif task == 2:
        M = data['Rotation_Task2']
        opt = data['GO_Task2']
        l, u, d = (-0.5, 0.5, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Weierstrass(x, M, opt)

def NILS(array, task, data):
    if task == 1:
        M = data['Rotation_Task1']
        opt = data['GO_Task1']
        l, u, d = (-50, 50, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Rastrigin(x, M, opt)
    elif task == 2:
        l, u, d = (-500, 500, 50)
        x = l + array * (u - l)
        x = x[:d]
        return Schwefel(x)
BENCHMARK_FILES = {'CIHS': 'CI_H.mat', 'CIMS': 'CI_M.mat', 'CILS': 'CI_L.mat', 'PIHS': 'PI_H.mat', 'PIMS': 'PI_M.mat', 'PILS': 'PI_L.mat', 'NIHS': 'NI_H.mat', 'NIMS': 'NI_M.mat', 'NILS': 'NI_L.mat'}

def get_data(question):
    """Read benchmark data independently of the current working directory."""
    if question not in BENCHMARK_FILES:
        raise ValueError(f'Unknown benchmark: {question!r}')
    path = Path(__file__).resolve().parent / 'Tasks' / BENCHMARK_FILES[question]
    if not path.is_file():
        raise FileNotFoundError(f'Missing benchmark data: {path}. See Tasks/README.md.')
    return sio.loadmat(path)

class Tasks:

    def __init__(self, question, task):
        self.question = question
        self.task = task
        self.data = get_data(self.question)

    def function(self, array):
        if self.question == 'CIHS':
            fitness = CIHS(array, self.task, self.data)
        elif self.question == 'CIMS':
            fitness = CIMS(array, self.task, self.data)
        elif self.question == 'CILS':
            fitness = CILS(array, self.task, self.data)
        elif self.question == 'PIHS':
            fitness = PIHS(array, self.task, self.data)
        elif self.question == 'PIMS':
            fitness = PIMS(array, self.task, self.data)
        elif self.question == 'PILS':
            fitness = PILS(array, self.task, self.data)
        elif self.question == 'NIHS':
            fitness = NIHS(array, self.task, self.data)
        elif self.question == 'NIMS':
            fitness = NIMS(array, self.task, self.data)
        elif self.question == 'NILS':
            fitness = NILS(array, self.task, self.data)
        return fitness
