// Import libraries
const Decimal = require('decimal.js');
const { symbols } = require('sympy');
const heapq = require('heapq');
const math = require('mathjs');
const crypto = require('crypto');
const np = require('numjs');
const mp = require('mpmath');

// Define factorial function
export function factorial(num) {
    if (num < 0) {
        throw new Error("Factorial is not defined for negative numbers");
    }
    let result = 1;
    for (let i = 1; i <= num; i++) {
        result *= i;
    }
    return result;
}

// Define fibonacci function
export function fibonacci(n) {
    if (n < 0) {
        throw new Error("Fibonacci number is not defined for negative indices");
    }
    if (n <= 1) {
        return n;
    }
    let fib_sequence = [0, 1];
    for (let i = 2; i <= n; i++) {
        let next_num = fib_sequence[i - 1] + fib_sequence[i - 2];
        fib_sequence.push(next_num);
    }
    return fib_sequence[n];
}

// Define power function
export function power(base, exponent) {
    if (exponent === 0) {
        return 1;
    }
    if (exponent < 0) {
        base = 1 / base;
        exponent = -exponent;
    }
    let result = 1;
    while (exponent > 0) {
        if (exponent % 2 === 1) {
            result *= base;
        }
        base *= base;
        exponent = Math.floor(exponent / 2);
    }
    if (Number.isInteger(result)) {
        return result;
    } else {
        return result.toFixed(10);
    }
}

// Define sum_of_numbers function
export function sum_of_numbers(n) {
    if (n < 1) {
        throw new Error("N must be positive");
    }
    let result = 0;
    for (let i = 1; i <= n; i++) {
        result += i;
    }
    return result;
}

// Define gcd function
export function gcd(a, b) {
    while (b) {
        [a, b] = [b, a % b];
    }
    return a;
}

// Define binomial function
export function binomial(n, k) {
    if (k < 0 || k > n) {
        return 0;
    }
    let result = 1;
    for (let i = 1; i <= Math.min(k, n - k); i++) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

// Define calculate_pi function
export function calculate_pi(num_digits) {
    Decimal.set({ precision: num_digits + 2 });
    const one_239 = new Decimal(1).div(239);
    let pi = new Decimal(0);
    for (let k = 0; k < num_digits; k++) {
        pi = pi.plus(new Decimal(1).div(16 ** k).times(
            new Decimal(4).div(8 * k + 1).minus(new Decimal(2).div(8 * k + 4)).minus(new Decimal(1).div(8 * k + 5)).minus(new Decimal(1).div(8 * k + 6))
        ));
        pi = pi.minus(one_239.div(8 ** k));
    }
    return pi.toFixed(num_digits);
}

// Define mathPi function
export function mathPi(num) {
    const num_digits = parseInt(num);
    if (num_digits < 0) {
        throw new Error("The number of characters must be a non-negative number");
    }
    const pi = calculate_pi(num_digits);
    return pi;
}

// Define square_root function
export function square_root(number) {
    if (number < 0) {
        throw new Error("The square root of a negative number is undefined");
    }
    return Math.sqrt(number);
}

// Define matrix_multiply function
export function matrix_multiply(matrix1, matrix2) {
    if (matrix1[0].length !== matrix2.length) {
        throw new Error("Matrix size mismatch");
    }
    const result = [];
    for (let i = 0; i < matrix1.length; i++) {
        result[i] = [];
        for (let j = 0; j < matrix2[0].length; j++) {
            result[i][j] = 0;
            for (let k = 0; k < matrix2.length; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

// Define matrix_determinant function
export function matrix_determinant(matrix) {
    if (matrix.length !== matrix[0].length) {
        throw new Error("The matrix must be square");
    }
    const n = matrix.length;
    if (n === 1) {
        return matrix[0][0];
    }
    if (n === 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    let det = 0;
    for (let j = 0; j < n; j++) {
        const minor = matrix.slice(1).map(row => row.slice(0, j).concat(row.slice(j + 1)));
        det += matrix[0][j] * matrix_determinant(minor) * (-1) ** j;
    }
    return det;
}

// Define matrix_inverse function
export function matrix_inverse(matrix) {
    if (matrix.length !== matrix[0].length) {
        throw new Error("The matrix must be square");
    }
    const n = matrix.length;
    const identity = [];
    for (let i = 0; i < n; i++) {
        identity[i] = [];
        for (let j = 0; j < n; j++) {
            identity[i][j] = i === j ? 1 : 0;
        }
    }
    for (let i = 0; i < n; i++) {
        const factor = matrix[i][i];
        if (factor === 0) {
            throw new Error("Matrix has no matrix inverse");
        }
        for (let j = 0; j < n; j++) {
            matrix[i][j] /= factor;
            identity[i][j] /= factor;
        }
        for (let k = 0; k < n; k++) {
            if (k === i) {
                continue;
            }
            const factor = matrix[k][i];
            for (let j = 0; j < n; j++) {
                matrix[k][j] -= factor * matrix[i][j];
                identity[k][j] -= factor * identity[i][j];
            }
        }
    }
    return identity;
}

// Define derivative function
export function derivative(expression) {
    return expression.diff(x);
}

// Define integral function
export function integral(expression) {
    return integrate(expression, x);
}

// Define sin function
export function sin(x) {
    return Math.sin(x);
}

// Define cos function
export function cos(x) {
    return Math.cos(x);
}

// Define tan function
export function tan(x) {
    return Math.tan(x);
}

// Define cotg function
export function cotg(x) {
    return 1 / Math.tan(x);
}

// Define sec function
export function sec(x) {
    return 1 / Math.cos(x);
}

// Define csc function
export function csc(x) {
    return 1 / Math.sin(x);
}

// Define asin function
export function asin(x) {
    return Math.asin(x);
}

// Define arcos function
export function arcos(x) {
    return Math.acos(x);
}

// Define artan function
export function artan(x) {
    return Math.atan(x);
}

// Define arcotg function
export function arcotg(x) {
    return Math.atan(1 / x);
}

// Define arsec function
export function arsec(x) {
    if (x >= 1 || x <= -1) {
        return Math.acos(1 / x);
    } else {
        throw new Error("asec(x) is undefined for |x| < 1");
    }
}

// Define arcsc function
export function arcsc(x) {
    if (x >= 1 || x <= -1) {
        return Math.asin(1 / x);
    } else {
        throw new Error("acsc(x) is undefined for |x| < 1");
    }
}

// Define sinh function
export function sinh(x) {
    return Math.sinh(x);
}

// Define cosh function
export function cosh(x) {
    return Math.cosh(x);
}

// Define tanh function
export function tanh(x) {
    return Math.tanh(x);
}

// Define asinh function
export function asinh(x) {
    return Math.asinh(x);
}

// Define acosh function
export function acosh(x) {
    if (x < 1) {
        throw new Error("Input to acosh must be >= 1");
    }
    return Math.acosh(x);
}

// Define atanh function
export function atanh(x) {
    if (Math.abs(x) >= 1) {
        throw new Error("Input to atanh must be < 1 in absolute value");
    }
    return Math.atanh(x);
}

// Define sec function
export function sec(x) {
    return 1 / Math.cos(x);
}

// Define csc function
export function csc(x) {
    return 1 / Math.sin(x);
}

// Define cot function
export function cot(x) {
    return 1 / Math.tan(x);
}

// Define asec function
export function asec(x) {
    if (x >= 1 || x <= -1) {
        return Math.acos(1 / x);
    } else {
        throw new Error("asec(x) is undefined for |x| < 1");
    }
}

// Define acsc function
export function acsc(x) {
    if (x >= 1 || x <= -1) {
        return Math.asin(1 / x);
    } else {
        throw new Error("acsc(x) is undefined for |x| < 1");
    }
}

// Define acot function
export function acot(x) {
    return Math.atan(1 / x);
}

// Define mean function
export function mean(numbers) {
    if (numbers.length === 0) {
        throw new Error("List of numbers is empty");
    }
    return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
}

// Define median function
export function median(numbers) {
    if (numbers.length === 0) {
        throw new Error("List of numbers is empty");
    }
    const sorted_numbers = numbers.sort((a, b) => a - b);
    const n = numbers.length;
    if (n % 2 === 0) {
        const mid1 = n / 2;
        const mid2 = mid1 - 1;
        return (sorted_numbers[mid1] + sorted_numbers[mid2]) / 2;
    } else {
        const mid = Math.floor(n / 2);
        return sorted_numbers[mid];
    }
}

// Define variance function
export function variance(numbers) {
    if (numbers.length === 0) {
        throw new Error("List of numbers is empty");
    }
    const n = numbers.length;
    const mean_value = mean(numbers);
    const squared_differences = numbers.map(num => (num - mean_value) ** 2);
    return squared_differences.reduce((sum, num) => sum + num, 0) / n;
}

// Define std_deviation function
export function std_deviation(numbers) {
    return Math.sqrt(variance(numbers));
}

// Define sieve_of_eratosthenes function
export function sieve_of_eratosthenes(limit) {
    if (limit < 2) {
        return [];
    }
    const is_prime = new Array(limit + 1).fill(true);
    is_prime[0] = is_prime[1] = false;
    for (let p = 2; p <= Math.floor(Math.sqrt(limit)); p++) {
        if (is_prime[p]) {
            for (let multiple = p * p; multiple <= limit; multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }
    const prime_numbers = [];
    for (let i = 2; i <= limit; i++) {
        if (is_prime[i]) {
            prime_numbers.push(i);
        }
    }
    return prime_numbers;
}

class TreeNode {
    constructor(key) {
        this.key = key;
        this.left = null;
        this.right = null;
    }
}

class BinaryTree {
    constructor() {
        this.root = null;
    }
    insert(key) {
        if (this.root === null) {
            this.root = new TreeNode(key);
        } else {
            this.root = this._insert(this.root, key);
        }
    }
    _insert(node, key) {
        if (node === null) {
            return new TreeNode(key);
        }
        if (key < node.key) {
            node.left = this._insert(node.left, key);
        } else {
            node.right = this._insert(node.right, key);
        }
        return node;
    }
    search(key) {
        return this._search(this.root, key);
    }
    _search(node, key) {
        if (node === null) {
            return null;
        }
        if (node.key === key) {
            return node;
        }
        if (key < node.key) {
            return this._search(node.left, key);
        }
        return this._search(node.right, key);
    }
}

class Graph {
    constructor(directed = false) {
        this.vertices = {};
        this.directed = directed;
    }
    add_vertex(vertex) {
        if (!(vertex in this.vertices)) {
            this.vertices[vertex] = [];
        }
    }
    add_edge(start, end, weight = 1) {
        if (start in this.vertices && end in this.vertices) {
            this.vertices[start].push([end, weight]);
            if (!this.directed) {
                this.vertices[end].push([start, weight]);
            }
        } else {
            throw new Error("One or both of the vertices do not exist in the graph.");
        }
    }
    get_neighbors(vertex) {
        return this.vertices[vertex] || [];
    }
    remove_vertex(vertex) {
        if (vertex in this.vertices) {
            delete this.vertices[vertex];
            for (let v in this.vertices) {
                this.vertices[v] = this.vertices[v].filter(([u, w]) => u !== vertex);
            }
        } else {
            throw new Error("Vertex does not exist in the graph.");
        }
    }
    remove_edge(start, end) {
        if (start in this.vertices && end in this.vertices) {
            this.vertices[start] = this.vertices[start].filter(([u, w]) => u !== end);
            if (!this.directed) {
                this.vertices[end] = this.vertices[end].filter(([u, w]) => u !== start);
            }
        } else {
            throw new Error("One or both of the vertices do not exist in the graph.");
        }
    }
    has_vertex(vertex) {
        return vertex in this.vertices;
    }
    has_edge(start, end) {
        if (start in this.vertices && end in this.vertices) {
            return this.vertices[start].some(([u, w]) => u === end);
        }
        return false;
    }
    remove_edges(vertex) {
        if (vertex in this.vertices) {
            for (let v in this.vertices) {
                this.vertices[v] = this.vertices[v].filter(([u, w]) => u !== vertex);
            }
        } else {
            throw new Error("Vertex does not exist in the graph.");
        }
    }
    num_vertices() {
        return Object.keys(this.vertices).length;
    }
    num_edges() {
        let edge_count = 0;
        for (let edges of Object.values(this.vertices)) {
            edge_count += edges.length;
        }
        return this.directed ? edge_count : edge_count / 2;
    }
    dfs(start, visited = new Set()) {
        if (!visited.has(start)) {
            visited.add(start);
            for (let [neighbor, _] of this.get_neighbors(start)) {
                this.dfs(neighbor, visited);
            }
        }
        return visited;
    }
    bfs(start) {
        const visited = new Set();
        const queue = [start];
        while (queue.length > 0) {
            const vertex = queue.shift();
            if (!visited.has(vertex)) {
                visited.add(vertex);
                const neighbors = this.get_neighbors(vertex).map(([neighbor, _]) => neighbor).filter(neighbor => !visited.has(neighbor));
                queue.push(...neighbors);
            }
        }
        return visited;
    }
    shortest_path_dijkstra(start, end) {
        const distances = {};
        for (let vertex in this.vertices) {
            distances[vertex] = Infinity;
        }
        distances[start] = 0;
        const priority_queue = new PriorityQueue();
        priority_queue.enqueue(0, start);
        while (!priority_queue.isEmpty()) {
            const [current_distance, current_vertex] = priority_queue.dequeue();
            if (current_distance > distances[current_vertex]) {
                continue;
            }
            for (let [neighbor, weight] of this.get_neighbors(current_vertex)) {
                const distance = current_distance + weight;
                if (distance < distances[neighbor]) {
                    distances[neighbor] = distance;
                    priority_queue.enqueue(distance, neighbor);
                }
            }
        }
        return distances[end];
    }
    is_connected() {
        if (Object.keys(this.vertices).length === 0) {
            throw new Error("Graph is empty, cannot determine connectivity.");
        }
        const start_vertex = Object.keys(this.vertices)[0];
        const visited = this.dfs(start_vertex);
        return visited.size === Object.keys(this.vertices).length;
    }
}

class PriorityQueue {
    constructor() {
        this.heap = [];
    }
    enqueue(priority, value) {
        heapq.heappush(this.heap, [priority, value]);
    }
    dequeue() {
        return heapq.heappop(this.heap);
    }
    isEmpty() {
        return this.heap.length === 0;
    }
}

export function is_sorted_descending(A) {
    const n = A.length;
    for (let i = 0; i < n - 1; i++) {
        if (A[i] < A[i + 1]) {
            return false;
        }
    }
    return true;
}

export function is_sorted(A) {
    const n = A.length;
    for (let i = 0; i < n - 1; i++) {
        if (A[i] > A[i + 1]) {
            return false;
        }
    }
    return true;
}

export function has_duplicates(nums) {
    let seen = 0;
    for (let num of nums) {
        if ((seen & (1 << num)) > 0) {
            return true;
        }
        seen |= 1 << num;
    }
    return false;
}

export function file_hash(file_path, hash_algorithm = "sha256") {
    if (!hashlib.getHashes().includes(hash_algorithm)) {
        throw new Error("Invalid hash algorithm");
    }
    const hash_obj = hashlib.createHash(hash_algorithm);
    const fs = require('fs');
    const file = fs.readFileSync(file_path);
    hash_obj.update(file);
    return hash_obj.digest('hex');
}

export function has_duplicate_values(lst) {
    let seen = 0;
    for (let item of lst) {
        const item_mask = 1 << item;
        if ((seen & item_mask) !== 0) {
            return true;
        }
        seen |= item_mask;
    }
    return false;
}

export function calculate_ln2(precision) {
    mp.mp.dps = precision + 2;
    let ln2 = mp.mpf(0);
    for (let n = 1; n <= precision * 2; n++) {
        ln2 = ln2.plus(mp.mpf(1).div(n).times((-1) ** (n - 1)));
    }
    return ln2.toFixed(precision);
}

export function calculate_pi(precision) {
    mp.mp.dps = precision + 2;
    let pi = mp.mpf(4).times(mp.atan(mp.mpf(1).div(5))).minus(mp.atan(mp.mpf(1).div(239))).times(4);
    return pi.toFixed(precision);
}

export function calculate_e_with_precision(precision) {
    mp.mp.dps = precision + 2;
    let e = mp.mpf(1);
    let factorial = mp.mpf(1);
    for (let i = 1; i <= precision * 2; i++) {
        factorial = factorial.times(i);
        e = e.plus(mp.mpf(1).div(factorial));
    }
    return e.toFixed(precision);
}

export function golden_ratio(precision) {
    mp.mp.dps = precision + 2;
    const phi = mp.mpf(1).plus(mp.sqrt(5)).div(2);
    return phi.toFixed(precision);
}

export function seidel_constant(A, tol = 1e-6, max_iterations = 1000) {
    const n = A.length;
    const x = new Array(n).fill(0);
    const L = np.tril(A, -1);
    const U = np.triu(A, 1);
    for (let i = 0; i < max_iterations; i++) {
        const x_new = new Array(n).fill(0);
        for (let j = 0; j < n; j++) {
            x_new[j] = (A[j][n - 1] - np.dot(U[j], x) - np.dot(L[j], x_new)) / A[j][j];
        }
        const error_norm = np.linalg.norm(np.subtract(x_new, x), np.inf) / (np.linalg.norm(x_new, np.inf) + 1e-20);
        x = x_new;
        if (error_norm < tol) {
            const eigenvalues_L = np.linalg.eigvals(L);
            const eigenvalues_U = np.linalg.eigvals(U);
            const max_eigenvalue_L = np.max(np.real(eigenvalues_L));
            const max_eigenvalue_U = np.max(np.real(eigenvalues_U));
            return 1 / (1 - max_eigenvalue_L / max_eigenvalue_U);
        }
    }
    throw new Error("Seidel constant did not converge within the specified number of iterations.");
}

export function log(x, base) {
    if (x <= 0 || base <= 0 || base === 1) {
        throw new Error("The number and base must be positive, and the base must not be 1.");
    }
    let result = 0;
    while (x >= base) {
        x /= base;
        result += 1;
    }
    return result;
}

export function tetration(base, n) {
    let result = 1;
    while (n > 0) {
        result = base ** result;
        n -= 1;
    }
    return result;
}
