# Graph and Linear Algebra Library

This library provides implementations for graph data structures and algorithms, as well as linear algebra operations. It includes functions for creating and manipulating directed and undirected graphs, performing depth-first and breadth-first searches, finding shortest paths using Dijkstra's algorithm, and checking graph properties like connectivity.

## Graph Data Structure

### `Graph` Struct

The `Graph` struct represents a graph with vertices and edges. It contains an adjacency list to store the vertices and their corresponding edges.

#### Functions:

- `createGraph(directed)`: Creates and initializes a new graph.

    Parameters:
    - `directed`: A boolean indicating whether the graph is directed or undirected.

- `addVertex(graph, vertex)`: Adds a vertex to the graph.

    Parameters:
    - `graph`: The graph to which the vertex will be added.
    - `vertex`: The vertex to be added.

- `addEdge(graph, start, end, weight)`: Adds an edge between two vertices with an optional weight.

    Parameters:
    - `graph`: The graph to which the edge will be added.
    - `start`: The starting vertex of the edge.
    - `end`: The ending vertex of the edge.
    - `weight`: The weight of the edge.

- `getNeighbors(graph, vertex, numNeighbors)`: Returns an array of neighbors for a given vertex.

    Parameters:
    - `graph`: The graph from which neighbors will be retrieved.
    - `vertex`: The vertex for which neighbors are requested.
    - `numNeighbors`: Pointer to an integer to store the number of neighbors.

- `removeVertex(graph, vertex)`: Removes a vertex and its associated edges from the graph.

    Parameters:
    - `graph`: The graph from which the vertex will be removed.
    - `vertex`: The vertex to be removed.

- `removeEdge(graph, start, end)`: Removes an edge between two vertices.

    Parameters:
    - `graph`: The graph from which the edge will be removed.
    - `start`: The starting vertex of the edge.
    - `end`: The ending vertex of the edge.

- `hasVertex(graph, vertex)`: Checks if a vertex exists in the graph.

    Parameters:
    - `graph`: The graph to be checked.
    - `vertex`: The vertex to be checked.

- `hasEdge(graph, start, end)`: Checks if an edge exists between two vertices.

    Parameters:
    - `graph`: The graph to be checked.
    - `start`: The starting vertex of the edge.
    - `end`: The ending vertex of the edge.

- `removeEdges(graph, vertex)`: Removes all edges connected to a given vertex.

    Parameters:
    - `graph`: The graph from which edges will be removed.
    - `vertex`: The vertex for which edges will be removed.

- `numVertices(graph)`: Returns the number of vertices in the graph.

- `numEdges(graph)`: Returns the number of edges in the graph.

- `dfs(graph, start, visited, numVisited)`: Performs depth-first search starting from a given vertex.

    Parameters:
    - `graph`: The graph on which DFS will be performed.
    - `start`: The starting vertex for DFS.
    - `visited`: An array to mark visited vertices.
    - `numVisited`: Pointer to an integer to store the number of visited vertices.

- `depthFirstSearch(graph, start, numVisited)`: Returns an array indicating which vertices are reachable from a given vertex using depth-first search.

    Parameters:
    - `graph`: The graph on which DFS will be performed.
    - `start`: The starting vertex for DFS.
    - `numVisited`: Pointer to an integer to store the number of visited vertices.

- `breadthFirstSearch(graph, start, numVisited)`: Returns an array indicating which vertices are reachable from a given vertex using breadth-first search.

    Parameters:
    - `graph`: The graph on which BFS will be performed.
    - `start`: The starting vertex for BFS.
    - `numVisited`: Pointer to an integer to store the number of visited vertices.

- `shortestPathDijkstra(graph, start, end)`: Finds the shortest path between two vertices using Dijkstra's algorithm.

    Parameters:
    - `graph`: The graph in which the path will be found.
    - `start`: The starting vertex of the path.
    - `end`: The ending vertex of the path.

- `isConnected(graph)`: Checks if the graph is connected.

    Parameters:
    - `graph`: The graph to be checked.

## Linear Algebra Operations

### `seidelConstant` Function

The `seidelConstant` function calculates the Seidel constant for a given matrix using the Gauss-Seidel method.

#### Function:

- `seidelConstant(A, n, tol, maxIterations)`: Calculates the Seidel constant for a given matrix.

    Parameters:
    - `A`: The matrix for which the Seidel constant will be

 calculated.
    - `n`: The size of the matrix.
    - `tol`: The tolerance for convergence.
    - `maxIterations`: The maximum number of iterations.

### `powerIteration` Function

The `powerIteration` function performs the power iteration method to find the dominant eigenvalue and eigenvector of a matrix.

#### Function:

- `powerIteration(A, n, tol, maxIterations)`: Performs power iteration to find the dominant eigenvalue and eigenvector of a matrix.

    Parameters:
    - `A`: The matrix for which the dominant eigenvalue and eigenvector will be calculated.
    - `n`: The size of the matrix.
    - `tol`: The tolerance for convergence.
    - `maxIterations`: The maximum number of iterations.

## Build and Run

To build and run the library, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/graph-linear-algebra-library.git
   ```

2. Compile the library:

   ```bash
   cd graph-linear-algebra-library
   make
   ```

3. Run the example program:

   ```bash
   ./example
   ```

## Author
[KnightBits](https://github.com/KnightBits)