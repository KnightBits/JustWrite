from decimal import Decimal, getcontext
from sympy import symbols
import heapq, math
import hashlib
import numpy as np
from mpmath import mp

def factorial(num):
    if num < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(1, num + 1):
        result *= i
    return result

def fibonacci(n):
    if n < 0:
        raise ValueError("Fibonacci number is not defined for negative indices")
    if n <= 1:
        return n
    fib_sequence = [0, 1]
    for i in range(2, n + 1):
        next_num = fib_sequence[i - 1] + fib_sequence[i - 2]
        fib_sequence.append(next_num)
    return fib_sequence[n]

def power(base, exponent):
    if exponent == 0:
        return 1
    
    if exponent < 0:
        base = 1 / base
        exponent = -exponent
    
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base
        base *= base
        exponent //= 2
    
    if isinstance(result, int):
        return result
    else:
        return "{:.10f}".format(result)

def sum_of_numbers(n):
    if n < 1:
        raise ValueError("N must be positive")
    result = 0
    for i in range(1, n + 1):
        result += i
    return result

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def binomial(n, k):
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(1, min(k, n - k) + 1):
        result = result * (n - i + 1) // i
    return result

def calculate_pi(num_digits):
    getcontext().prec = num_digits + 2
    one_239 = Decimal(1) / 239
    pi = Decimal(0)
    for k in range(num_digits):
        pi += (Decimal(1) / 16 ** k) * (
            Decimal(4) / (8 * k + 1) - Decimal(2) / (8 * k + 4) - Decimal(1) / (8 * k + 5) - Decimal(1) / (8 * k + 6)
        )
        pi -= one_239 / 8 ** k
    return str(pi)[:-1]

def mathPi(num):
    num_digits = int(num)
    if num_digits < 0:
        raise ValueError("The number of characters must be a non-negative number")
    pi = calculate_pi(num_digits)
    return pi

import math

def square_root(number):
    if number < 0:
        raise ValueError("The square root of a negative number is undefined")
    return number**0.5

def matrix_multiply(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrix size mismatch")
    
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result

def matrix_determinant(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix must be square")
    
    n = len(matrix)
    
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += matrix[0][j] * matrix_determinant(minor) * (-1) ** j
    
    return det

def matrix_inverse(matrix):
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix must be square")
    
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    for i in range(n):
        factor = matrix[i][i]
        if factor == 0:
            raise ValueError("Matrix has no matrix inverse")
        
        for j in range(n):
            matrix[i][j] /= factor
            identity[i][j] /= factor
        
        for k in range(n):
            if k == i:
                continue
            factor = matrix[k][i]
            for j in range(n):
                matrix[k][j] -= factor * matrix[i][j]
                identity[k][j] -= factor * identity[i][j]
    
    return identity

x = symbols('x')

# производная
def derivative(expression):
    return expression.diff(x)
# интеграл
def integral(expression):
    return integrate(expression, x)

# синус
def sin(x):
    return math.sin(x)
# косинус
def cos(x):
    return math.cos(x)
# тангенс
def tan(x):
    return math.tan(x)
# котанкенс
def cotg(x):
    return 1 / math.tan(x)
# секанс
def sec(x):
    return 1 / math.cos(x)
# косеканс
def csc(x):
    return 1 / math.sin(x)

def asin(x):
    return math.asin(x)

def arcos(x):
    return math.acos(x)

def artan(x):
    return math.atan(x)

def arcotg(x):
    return math.atan(1 / x)

def arsec(x):
    if x >= 1 or x <= -1:
        return math.acos(1 / x)
    else:
        raise ValueError("asec(x) is undefined for |x| < 1")

def arcsc(x):
    if x >= 1 or x <= -1:
        return math.asin(1 / x)
    else:
        raise ValueError("acsc(x) is undefined for |x| < 1")

def sinh(x):
    return math.sinh(x)

def cosh(x):
    return math.cosh(x)

def tanh(x):
    return math.tanh(x)

# Обратные гиперболические функции
def asinh(x):
    return math.asinh(x)

def acosh(x):
    if x < 1:
        raise ValueError("Input to acosh must be >= 1")
    return math.acosh(x)

def atanh(x):
    if abs(x) >= 1:
        raise ValueError("Input to atanh must be < 1 in absolute value")
    return math.atanh(x)

# Тригонометрические функции
def sec(x):
    return 1 / math.cos(x)

def csc(x):
    return 1 / math.sin(x)

def cot(x):
    return 1 / math.tan(x)

# Обратные тригонометрические функции
def asec(x):
    if x >= 1 or x <= -1:
        return math.acos(1 / x)
    else:
        raise ValueError("asec(x) is undefined for |x| < 1")

def acsc(x):
    if x >= 1 or x <= -1:
        return math.asin(1 / x)
    else:
        raise ValueError("acsc(x) is undefined for |x| < 1")

def acot(x):
    return math.atan(1 / x)

# Статистические функции

# среднее-арифмитическое

def mean(numbers):
    if len(numbers) == 0:
        raise ValueError("List of numbers is empty")
    return sum(numbers) / len(numbers)

# медиана

def median(numbers):
    if len(numbers) == 0:
        raise ValueError("List of numbers is empty")
    sorted_numbers = sorted(numbers)
    n = len(numbers)
    if n % 2 == 0:
        mid1 = n // 2
        mid2 = mid1 - 1
        return (sorted_numbers[mid1] + sorted_numbers[mid2]) / 2
    else:
        mid = n // 2
        return sorted_numbers[mid]

# дипрессия

def variance(numbers):
    if len(numbers) == 0:
        raise ValueError("List of numbers is empty")
    n = len(numbers)
    mean_value = mean(numbers)
    squared_differences = [(x - mean_value) ** 2 for x in numbers]
    return sum(squared_differences) / n

# стандартное отклонение

def std_deviation(numbers):
    return math.sqrt(variance(numbers))

def sieve_of_eratosthenes(limit):
    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False

    prime_numbers = [i for i in range(2, limit + 1) if is_prime[i]]
    return prime_numbers

# binary treeclass TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return TreeNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        return node

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return None
        if node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)
# end of tree

#graphclass Graph:
    def __init__(self, directed=False):
        self.vertices = {}
        self.directed = directed

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, start, end, weight=1):
        if start in self.vertices and end in self.vertices:
            self.vertices[start].append((end, weight))
            if not self.directed:
                self.vertices[end].append((start, weight))
        else:
            raise ValueError("One or both of the vertices do not exist in the graph.")

    def get_neighbors(self, vertex):
        return self.vertices.get(vertex, [])

    def remove_vertex(self, vertex):
        if vertex in self.vertices:
            del self.vertices[vertex]
            for v in self.vertices:
                self.vertices[v] = [(u, w) for u, w in self.vertices[v] if u != vertex]
        else:
            raise ValueError("Vertex does not exist in the graph.")

    def remove_edge(self, start, end):
        if start in self.vertices and end in self.vertices:
            self.vertices[start] = [(u, w) for u, w in self.vertices[start] if u != end]
            if not self.directed:
                self.vertices[end] = [(u, w) for u, w in self.vertices[end] if u != start]
        else:
            raise ValueError("One or both of the vertices do not exist in the graph.")

    def has_vertex(self, vertex):
        return vertex in self.vertices

    def has_edge(self, start, end):
        if start in self.vertices and end in self.vertices:
            return end in [v for v, _ in self.vertices[start]]
        return False

    def remove_edges(self, vertex):
        if vertex in self.vertices:
            for v in self.vertices:
                self.vertices[v] = [(u, w) for u, w in self.vertices[v] if u != vertex]
        else:
            raise ValueError("Vertex does not exist in the graph.")

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        edge_count = sum(len(edges) for edges in self.vertices.values())
        return edge_count if self.directed else edge_count // 2

    def __str__(self):
        result = "Graph (Directed: {})\n".format(self.directed)
        for vertex, edges in self.vertices.items():
            result += "{} -> {}\n".format(vertex, [edge[0] for edge in edges])
        return result

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        if start not in visited:
            visited.add(start)
            for neighbor, _ in self.get_neighbors(start):
                self.dfs(neighbor, visited)
        return visited

    def bfs(self, start):
        visited = set()
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                neighbors = [neighbor for neighbor, _ in self.get_neighbors(vertex) if neighbor not in visited]
                queue.extend(neighbors)
        return visited

    def shortest_path_dijkstra(self, start, end):
        distances = {vertex: float('infinity') for vertex in self.vertices}
        distances[start] = 0

        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor, weight in self.get_neighbors(current_vertex):
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[end]

    def is_connected(self):
        if not self.vertices:
            raise ValueError("Graph is empty, cannot determine connectivity.")
        start_vertex = list(self.vertices.keys())[0]
        visited = self.dfs(start_vertex)
        return len(visited) == len(self.vertices)

# проверка, отсортирован ли список по убыванию (True \ False)

def is_sorted_descending(A):
    n = len(A)
    for i in range(n - 1):
        if A[i] < A[i + 1]:
            return False
    return True

# проверка, отсортирован ли список по возрастанию (True \ False)

def is_sorted(A):
    n = len(A)
    for i in range(n - 1):
        if A[i] > A[i + 1]:
            return False
    return True

# проверка на наличие дубликатов в списке

def has_duplicates(nums):
    seen = 0

    for num in nums:
        if (seen & (1 << num)) > 0:
            return True
        seen |= 1 << num

    return False

# хэш-сумма файла

def file_hash(file_path, hash_algorithm="sha256"):
    try:
        if hash_algorithm not in hashlib.algorithms_guaranteed:
            raise ValueError("Invalid hash algorithm")
        hash_obj = hashlib.new(hash_algorithm)

        with open(file_path, "rb") as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)

        return hash_obj.hexdigest()
    except Exception as e:
        return str(e)

# поиск дубликатов в списке с помощью битовых операций

def has_duplicate_values(lst):
    seen = 0

    for item in lst:
        item_mask = 1 << item
        if (seen & item_mask) != 0:
            return True
        seen |= item_mask

    return False

# вычисление числа натурального логарифма 2 (ln(2))

def calculate_ln2(precision):
    getcontext().prec = precision + 2
    ln2 = Decimal(0)
    for n in range(1, precision * 2):
        ln2 += 1 / n * ((-1) ** (n - 1))
    return str(ln2)[:-1]

# рассчеты числа пи
def calculate_pi(precision):
    getcontext().prec = precision + 2
    pi = Decimal(4) * (4 * Decimal(math.atan(1/5)) - Decimal(math.atan(1/239)))
    return str(pi)[:-1]


# экспонента е

def calculate_e_with_precision(precision):
    getcontext().prec = precision + 2
    e = Decimal(1)
    factorial = Decimal(1)
    for i in range(1, precision * 2):
        factorial *= i
        e += 1 / factorial
    return str(e)[:-1]

# число фи

def golden_ratio(precision):
    getcontext().prec = precision + 2
    phi = (1 + Decimal(5).sqrt()) / 2
    return str(phi)[:-1]

# константа Зейделя

def seidel_constant(A, tol=1e-6, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)
    
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    
    for i in range(max_iterations):
        x_new = np.zeros(n)
        for j in range(n):
            x_new[j] = (A[j, -1] - np.dot(U[j], x) - np.dot(L[j], x_new)) / A[j, j]
        
        error_norm = np.linalg.norm(x_new - x, np.inf) / (np.linalg.norm(x_new, np.inf) + 1e-20)
        
        x = x_new
        
        if error_norm < tol:
            eigenvalues_L = np.linalg.eigvals(L)
            eigenvalues_U = np.linalg.eigvals(U)
            max_eigenvalue_L = np.max(np.real(eigenvalues_L))
            max_eigenvalue_U = np.max(np.real(eigenvalues_U))
            
            return 1 / (1 - max_eigenvalue_L / max_eigenvalue_U)
    
    raise Exception("Seidel constant did not converge within the specified number of iterations.")

def log(x, base):
    if x <= 0 or base <= 0 or base == 1:
        raise ValueError("The number and base must be positive, and the base must not be 1.")
    result = 0
    while x >= base:
        x /= base
        result += 1
    return result

def tetration(base, n):
    result = 1
    while n > 0:
        result = base ** result
        n -= 1
    return result