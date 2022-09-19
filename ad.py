import numpy as np
from enum import IntEnum
from utils import print_tree

global_variable_counter = 0


class Operation(IntEnum):
    S_ADD = 0
    S_SUB = 1
    S_MUL = 2
    S_DIV = 3
    M_ADD = 4
    M_SUB = 5
    M_MUL = 6
    M_ELM = 7
    M_ELD = 8
    S_SQR = 9
    M_ESQ = 10
    M_RDS = 11
    M_TNP = 12


S_OPS = {Operation.S_ADD, Operation.S_SUB, Operation.S_MUL, Operation.S_DIV}
M_ELE = {Operation.M_ADD, Operation.M_SUB, Operation.M_ELM, Operation.M_ELD}
names = ['add', 'subtract', 'multiply', 'divide', 'add', 'subtract', 'multiply', 'multiply', 'divide', 'square root']
specific_names = ['Scalar Addition', 'Scalar Subtraction', 'Scalar Multiplication', 'Scalar Division', 'Matrix Addition', 'Matrix Subtraction', 'Matrix Multiplication', 'Matrix Element-wise Multiplication', 'Matrix Division', 'Scalar Square', 'Matrix Element-wise Square', 'Matrix Reduce Sum']

names_to_ops = {'add': [Operation.S_ADD, Operation.M_ADD],
                'subtract': [Operation.S_SUB, Operation.M_SUB],
                'multiply': [Operation.S_MUL, Operation.M_ELM],
                'divide': [Operation.S_DIV, Operation.M_ELD],
                'square': [Operation.S_SQR, Operation.M_ESQ]
}


class Variable:
    def __init__(self, operation: Operation = None, value=None, name=None, a=None, b=None, trainable=True):
        self.trainable = trainable
        if name:
            self.name = name
        else:
            global global_variable_counter
            self.name = f"Variable {global_variable_counter}"
            global_variable_counter += 1

        if value is not None:
            if type(value) == np.ndarray:
                self.value = value
                self.shape = value.shape
            elif type(value) == float:
                self.value = value
                self.shape = ()
            elif type(value) == list:
                self.value = np.array(value)
                self.shape = self.value.shape
            else:
                raise TypeError("Supplied value is not of type np.ndarray, list, or float")

            if len(self.shape) > 2:
                raise ValueError("Cannot create tensors of greater than 2 dimensions")

            self.op = None
            self.a = None
            self.b = None

            return
        else:
            self.value = value

        self.op = operation

        if self.op in S_OPS:
            if not (a.is_scalar() and b.is_scalar()):
                raise ValueError("One or both operands is not a scalar")
            self.a = a
            self.b = b
            self.shape = ()
        elif self.op in M_ELE:
            if not a.is_matrix() or not b.is_matrix():
                raise ValueError("One or both operands is not a matrix")
            if not a.shape == b.shape:
                raise ValueError("Matrices are not of same shape")
            self.a = a
            self.b = b
            self.shape = a.shape
        elif self.op == Operation.M_MUL:
            if not len(a.shape) > 0 or not len(b.shape) > 0:
                raise ValueError("One or both operands is not a matrix")
            if not b.shape[1] == 1:
                raise ValueError("Only matrix * column vector is supported for differentiation")
            if not a.shape[1] == b.shape[0]:
                raise ValueError("Incompatible shapes for matrix multiplication")
            self.a = a
            self.b = b
            self.shape = (a.shape[0], b.shape[1])
        elif self.op == Operation.S_SQR:
            if not a.is_scalar():
                raise ValueError("Operand is not a scalar")
            self.a = a
            self.b = None
            self.shape = ()
        elif self.op == Operation.M_ESQ:
            if not a.is_matrix():
                raise ValueError("Operand is not a matrix")
            self.a = a
            self.b = None
            self.shape = self.a.shape
        elif self.op == Operation.M_RDS:
            if not a.is_matrix():
                raise ValueError("Operand is not a matrix")
            self.a = a
            self.b = None
            self.shape = ()
        elif self.op == Operation.M_TNP:
            if not a.is_matrix():
                raise ValueError("Operand is not a matrix")
            self.a = a
            self.b = None
            self.shape = (self.a.shape[1], self.a.shape[0])
        else:
            raise ValueError("Something has gone horribly wrong!")

    def __str__(self):
        if self.a is not None:
            return f"Placeholder(\"{specific_names[int(self.op)]}\" shape={self.shape})"
        else:
            return f"Variable(\"{self.value}\")"

    def is_scalar(self):
        return self.shape == ()

    def is_matrix(self):
        return not self.is_scalar()

    def eval(self):
        if self.op is None:
            return self.value
        elif self.op == Operation.S_ADD:
            self.value = self.a.eval() + self.b.eval()
        elif self.op == Operation.S_SUB:
            self.value = self.a.eval() - self.b.eval()
        elif self.op == Operation.S_MUL:
            self.value = self.a.eval() * self.b.eval()
        elif self.op == Operation.S_DIV:
            self.value = self.a.eval() / self.b.eval()
        elif self.op == Operation.S_SQR:
            self.value = self.a.eval() * self.a.eval()
        elif self.op == Operation.M_ADD:
            self.value = self.a.eval() * self.b.eval()
        elif self.op == Operation.M_SUB:
            self.value = self.a.eval() - self.b.eval()
        elif self.op == Operation.M_ELM:
            self.value = self.a.eval() * self.b.eval()
        elif self.op == Operation.M_ELD:
            self.value = self.a.eval() / self.b.eval()
        elif self.op == Operation.M_ESQ:
            self.value = np.square(self.a.eval())
        elif self.op == Operation.M_MUL:
            self.value = np.matmul(self.a.eval(), self.b.eval())
        elif self.op == Operation.M_RDS:
            self.value = np.sum(self.a.eval())
        elif self.op == Operation.M_TNP:
            self.value = self.a.eval().T
        else:
            print("something has gone horribly wrong")
        return self.value

    def differentiate(self, a_or_b, d_du):
        if not (a_or_b == 0 or a_or_b == 1):
            raise ValueError("a_or_b must be either 0 or 1")
        if self.op is None:
            return None
        elif self.op == Operation.S_ADD:
            return d_du * np.float32(1.)
        elif self.op == Operation.S_SUB:
            return d_du * np.float32(-1.)
        elif self.op == Operation.S_MUL:
            if a_or_b == 0:
                return d_du * self.b.value
            else:
                return d_du * self.a.value
        elif self.op == Operation.S_DIV:
            if a_or_b == 0:
                return d_du / self.b.value
            else:
                return -d_du * self.a.value / (self.b.value * self.b.value)
        elif self.op == Operation.S_SQR:
            if a_or_b == 1:
                raise ValueError("b is not a valid option for square operation")
            return 2 * d_du * self.a.value
        elif self.op == Operation.M_ADD:
            return d_du * np.ones_like(self.a.value)
        elif self.op == Operation.M_SUB:
            return d_du * np.ones_like(self.a.value)
        elif self.op == Operation.M_ELM:
            if a_or_b == 0:
                return d_du * self.b.value
            else:
                return d_du * self.a.value
        elif self.op == Operation.M_ELD:
            if a_or_b == 0:
                return d_du / self.b.value
            else:
                return -d_du * self.a.value / (self.b.value * self.b.value)
        elif self.op == Operation.M_ESQ:
            if a_or_b == 1:
                raise ValueError("b is not a valid option for matrix element-wise square operation")
            return 2 * d_du * self.a.value
        elif self.op == Operation.M_MUL:
            if a_or_b == 0:
                diag_d_du = np.diag(d_du)
                return np.matmul(diag_d_du, np.tile(self.b.value, (self.a.value.shape[0], 1)))
            else:
                return np.dot(d_du, self.a.value.T[0])
        elif self.op == Operation.M_RDS:
            return d_du * np.ones_like(self.a.value)
        elif self.op == Operation.M_TNP:
            return d_du.T
        else:
            print("something has gone horribly wrong")


    def s_or_ele_ops(self, other, op):
        if not type(other) == Variable:
            raise TypeError(f"Cannot {names[int(op)]} Variable to something else")
        if self.is_scalar():
            if not other.is_scalar():
                raise ValueError(f"Cannot {names[int(op)]} matrix to scalar")
            return Variable(names_to_ops[op][0], a=self, b=other, trainable=False, name=specific_names[int(names_to_ops[op][0])])
        else:
            if other.is_scalar():
                raise ValueError(f"Cannot {names[int(op)]} scalar to matrix")
            if not self.shape == other.shape:
                raise ValueError(f"Cannot {names[int(op)]} matrices of different shapes")
            return Variable(names_to_ops[op][1], a=self, b=other, trainable=False, name=specific_names[int(names_to_ops[op][1])])

    def __add__(self, other):
        return self.s_or_ele_ops(other, 'add')

    def __sub__(self, other):
        return self.s_or_ele_ops(other, 'subtract')

    def __mul__(self, other):
        return self.s_or_ele_ops(other, 'multiply')

    def __truediv__(self, other):
        return self.s_or_ele_ops(other, 'divide')

    def __matmul__(self, other):
        return Variable(Operation.M_MUL, a=self, b=other, trainable=False, name='matmul')


def square(x: Variable):
    if x.is_scalar():
        return Variable(Operation.S_SQR, a=x, trainable=False, name='square')
    else:
        return Variable(Operation.M_ESQ, a=x, trainable=False, name='square')


def reduce_sum(x: Variable):
    if x.is_scalar():
        return x
    else:
        return Variable(Operation.M_RDS, a=x, trainable=False, name='reduce sum')


def matmul(x1: Variable, x2: Variable):
    return x1 @ x2


def transpose(x: Variable):
    return Variable(Operation.M_TNP, a=x, trainable=False, name='transpose')


def MSE(y: Variable, y_: Variable):
    if not (y.shape == y_.shape):
        raise ValueError("Arguments must have same shape")

    return reduce_sum(square(y - y_))


def training_loop():
    X = np.ones((100, 5), dtype=np.float32)
    Y = np.zeros((100), dtype=np.float32)


def main():
    y = Variable(value=np.array([[1, 2, 3, 4]], dtype=np.float32), name='y')
    y_ = Variable(value=np.array([[1.1, 2.2, 3.3, 4.4]], dtype=np.float32), name='y_')

    x1 = Variable(value=np.array([[2., 2.], [1., 1.]], dtype=np.float32), name='x1')
    x2 = Variable(value=np.array([[1., 1., 1., 1.], [1.1, 1.1, 1.1, 1.1]], dtype=np.float32), name='x2')

    x3 = matmul(x1, x2)

    x4 = matmul(x3, transpose(y - y_))
    print(x4.eval())
    print_tree(x4, val='trainable')
    print(np.matmul(np.matmul(x1.value, x2.value), (y.value - y_.value).T))


if __name__ == "__main__":
    main()

