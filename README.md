# Python Math and Data Structures Library

This Python library contains a collection of mathematical functions and data structures for various purposes. These functions and classes are designed to be helpful for performing mathematical operations and working with fundamental data structures.

## Getting Started

To use this library, you need to include the necessary Python files in your project or script. You can simply copy and paste the functions and classes from this library into your own Python script or import them using the `import` statement.

```python
from math_data_structures import *
```

## Mathematical Functions

### Factorial

```python
factorial(num)
```

Calculates the factorial of a non-negative integer.

### Fibonacci

```python
fibonacci(n)
```

Calculates the nth Fibonacci number.

### Power

```python
power(base, exponent)
```

Computes the result of raising a base to a non-negative integer exponent.

### Sum of Numbers

```python
sum_of_numbers(n)
```

Calculates the sum of numbers from 1 to n.

### Greatest Common Divisor (GCD)

```python
gcd(a, b)
```

Finds the greatest common divisor of two non-negative integers.

### Binomial Coefficient

```python
binomial(n, k)
```

Computes the binomial coefficient "n choose k."

### Approximate Value of Pi

```python
mathPi(num)
```

Calculates an approximate value of π (pi) with a specified number of decimal places.

### Square Root

```python
square_root(number)
```

Calculates the square root of a non-negative number.

### Matrix Operations

#### Matrix Multiplication

```python
matrix_multiply(matrix1, matrix2)
```

Multiplies two matrices if their dimensions are compatible.

#### Matrix Determinant

```python
matrix_determinant(matrix)
```

Calculates the determinant of a square matrix.

#### Matrix Inverse

```python
matrix_inverse(matrix)
```

Finds the inverse of a square matrix.

### Calculus Operations

#### Compute Derivative

```python
compute_derivative(expression)
```

Computes the derivative of a mathematical expression with respect to the variable x.

#### Compute Integral

```python
compute_integral(expression)
```

Computes the integral of a mathematical expression with respect to the variable x.

### Trigonometric Functions

```python
sine(angle_in_radians)
cosine(angle_in_radians)
tangent(angle_in_radians)
```

Calculates the sine, cosine, and tangent of an angle in radians.

## Prime Number Generation

### Sieve of Eratosthenes

```python
sieve_of_eratosthenes(limit)
```

Generates prime numbers up to a specified limit using the Sieve of Eratosthenes algorithm.

## Data Structures

### Binary Tree

A simple binary tree data structure is provided for basic tree operations.

```python
# Create a binary tree
tree = BinaryTree()

# Insert a key into the tree
tree.insert(key)

# Search for a key in the tree
node = tree.search(key)
```

### Graph

A graph data structure is provided for creating, modifying, and traversing graphs.

```python
# Create an undirected graph
graph = Graph()

# Add vertices to the graph
graph.add_vertex(vertex)

# Add edges to the graph
graph.add_edge(start, end, weight)

# Check if the graph is connected
connected = graph.is_connected()

# Perform graph traversal (DFS and BFS)
visited_nodes = graph.dfs(start)
visited_nodes = graph.bfs(start)

# Find the shortest path using Dijkstra's algorithm
shortest_distance = graph.shortest_path_dijkstra(start, end)
```
## Author
[KnightBits](https://github.com/KnightBits)
