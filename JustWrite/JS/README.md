## Mathematics and Algorithms Library

This JavaScript library provides a collection of mathematical functions and algorithms for various mathematical computations, ranging from basic operations to advanced mathematical concepts. Additionally, it includes data structures and algorithms for graphs and numerical computing.

### Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Functions and Classes](#functions-and-classes)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)

### Introduction

This library is designed to assist developers in performing mathematical computations efficiently and accurately. It covers a wide range of topics, including basic arithmetic operations, number theory, calculus, linear algebra, and graph theory.

### Installation

To use this library in your project, you can install it via npm:

```bash
npm install math-algorithms-library
```

### Usage

To use a specific function or class from the library, you can import it as follows:

```javascript
// Import the entire library
const mathLibrary = require('math-algorithms-library');

// Import a specific function or class
const { factorial, fibonacci, Matrix } = require('math-algorithms-library');
```

### Functions and Classes

#### Basic Arithmetic and Number Theory
- `factorial(num)`: Computes the factorial of a non-negative integer.
- `fibonacci(n)`: Computes the n-th Fibonacci number.
- Various mathematical functions such as `power`, `sum_of_numbers`, `gcd`, `binomial`, etc.

#### Numerical Computing
- Matrix operations: `matrix_multiply`, `matrix_determinant`, `matrix_inverse`.
- Numerical constants: `calculate_pi`, `calculate_e_with_precision`, `golden_ratio`.
- Special functions: `seidel_constant`, `log`, `tetration`.

#### Trigonometric and Hyperbolic Functions
- Trigonometric functions: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`.
- Inverse trigonometric functions: `asin`, `arcos`, `artan`, `arcotg`, `arsec`, `arcsc`.

#### Statistical Functions
- `mean`, `median`, `variance`, `std_deviation`.

#### Graphs and Algorithms
- `Graph` class: Create and manipulate graphs, check connectivity, find shortest paths, etc.
- `BinaryTree` class: Implement a binary search tree.

#### Additional Utilities
- Various utility functions such as `is_sorted`, `has_duplicates`, `file_hash`, etc.

### Examples

Here are some examples of how to use the library:

#### Calculating Factorial

```javascript
const { factorial } = require('math-algorithms-library');

const result = factorial(5);
console.log(result); // Output: 120
```

#### Matrix Operations

```javascript
const { matrix_multiply, matrix_determinant, matrix_inverse } = require('math-algorithms-library');

const matrix1 = [[1, 2], [3, 4]];
const matrix2 = [[5, 6], [7, 8]];

const product = matrix_multiply(matrix1, matrix2);
console.log(product);

const det = matrix_determinant(matrix1);
console.log(det);

const inverse = matrix_inverse(matrix1);
console.log(inverse);
```

#### Graph Operations

```javascript
const { Graph } = require('math-algorithms-library');

const graph = new Graph();

graph.add_vertex('A');
graph.add_vertex('B');
graph.add_edge('A', 'B', 2);

const is_connected = graph.is_connected();
console.log(is_connected); // Output: true
```

### Contributing

Contributions to the library are welcome! If you have improvements or new features to suggest, feel free to create issues or submit pull requests.

## Author
[KnightBits](https://github.com/KnightBits)