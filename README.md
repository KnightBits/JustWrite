# Math and Data Structures Library

Welcome to the Math and Data Structures Library – your go-to resource for mathematical functions and essential data structures, designed to enhance your Python projects. This versatile library covers a wide range of mathematical operations and provides fundamental data structures for your computational needs.

## Overview

- **Mathematical Functions and Trigonometric Operations:** Explore a variety of mathematical functions and trigonometric operations for precise calculations.

  - `log(x, base)`: Calculate logarithms with a specified base.
  - `tetration(base, n)`: Perform tetration for powerful exponentiation.

- **Prime Number Generation:** Utilize efficient algorithms for generating prime numbers up to a specified limit.

  - `sieve_of_eratosthenes(limit)`: Apply the Sieve of Eratosthenes for prime number generation.

- **Factorial, Fibonacci, and More:** Access essential mathematical operations and series.

  - `factorial(num)`: Calculate the factorial of a non-negative integer.
  - `fibonacci(n)`: Discover the nth Fibonacci number.
  - `power(base, exponent)`: Compute the result of exponentiation.
  - `sum_of_numbers(n)`: Calculate the sum of numbers up to n.
  - `gcd(a, b)`: Find the greatest common divisor.
  - `binomial(n, k)`: Compute binomial coefficients.
  - `mathPi(num)`: Approximate the value of π (pi).
  - `square_root(number)`: Calculate the square root.

- **Matrix Operations:** Perform matrix operations with ease.

  - `matrix_multiply(matrix1, matrix2)`: Multiply two matrices.
  - `matrix_determinant(matrix)`: Calculate the determinant of a square matrix.
  - `matrix_inverse(matrix)`: Find the inverse of a square matrix.

- **Calculus Operations:** Dive into calculus with derivative and integral calculations.

  - `compute_derivative(expression)`: Calculate the derivative of a mathematical expression.
  - `compute_integral(expression)`: Compute the integral of a mathematical expression.

- **Trigonometric Functions:** Harness trigonometric functions for angle calculations.

  - `sine(angle_in_radians)`, `cosine(angle_in_radians)`, `tangent(angle_in_radians)`: Calculate trigonometric functions.

## Data Structures

- **Binary Tree:** Employ a simple binary tree for basic tree operations.

  ```python
  # Create a binary tree
  tree = BinaryTree()

  # Insert a key into the tree
  tree.insert(key)

  # Search for a key in the tree
  node = tree.search(key)
  ```

- **Graph:** Use a graph data structure for creating, modifying, and traversing graphs.

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
  visited_nodes_dfs = graph.dfs(start)
  visited_nodes_bfs = graph.bfs(start)

  # Find the shortest path using Dijkstra's algorithm
  shortest_distance = graph.shortest_path_dijkstra(start, end)
  ```

## Functions for Lists, Arrays, and More

- **Sorting and Duplicates:** Check sorting order and find duplicates in lists.

  - `is_sorted_descending(A)`: Check if a list is sorted in descending order.
  - `is_sorted(A)`: Check if a list is sorted in ascending order.
  - `has_duplicates(nums)`: Check for duplicates in a list.
  - `has_duplicate_values(lst)`: Find duplicates in a list using bitwise operations.

## Mathematical Calculations

- **Constant Values:** Calculate fundamental mathematical constants with precision.

  - `calculate_ln2(precision)`: Natural logarithm of 2 (ln(2)).
  - `calculate_pi(precision)`: Value of Pi (π).
  - `calculate_e_with_precision(precision)`: Euler's number (e).
  - `golden_ratio(precision)`: Golden ratio (phi).

## File Hash Calculations

- **Hash Algorithms:** Calculate file hashes using various hash algorithms.

  - `calculate_file_hash(file_path, hash_algorithm="sha256")`: Calculate file hash using a specified hash algorithm.

## Author

[KnightBits](https://github.com/KnightBits)

Dive into the power of mathematics and efficient data structures – your journey begins with the Math and Data Structures Library!
