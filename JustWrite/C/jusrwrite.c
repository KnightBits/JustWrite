#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct TreeNode {
    int key;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

typedef struct BinaryTree {
    TreeNode* root;
} BinaryTree;

TreeNode* createTreeNode(int key) {
    TreeNode* node = calloc(1, sizeof(TreeNode));
    if (!node) {
        fprintf(stderr, "Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    node->key = key;
    return node;
}

BinaryTree* createBinaryTree() {
    BinaryTree* tree = calloc(1, sizeof(BinaryTree));
    if (!tree) {
        fprintf(stderr, "Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    return tree;
}

TreeNode* insert(TreeNode* node, int key) {
    if (node == NULL) {
        return createTreeNode(key);
    }
    if (key < node->key) {
        node->left = insert(node->left, key);
    }
    else {
        node->right = insert(node->right, key);
    }
    return node;
}

void insertKey(BinaryTree* tree, int key) {
    tree->root = insert(tree->root, key);
}

TreeNode* search(const TreeNode* node, int key) {
    if (node == NULL || node->key == key) {
        return (TreeNode*)node;
    }
    if (key < node->key) {
        return search(node->left, key);
    }
    return search(node->right, key);
}

TreeNode* searchKey(const BinaryTree* tree, int key) {
    return search(tree->root, key);
}

typedef struct Edge {
    int end;
    int weight;
} Edge;

typedef struct Graph {
    int** vertices;
    int numVertices;
    int directed;
} Graph;

Graph* createGraph(int directed) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = NULL;
    graph->numVertices = 0;
    graph->directed = directed;
    return graph;
}

void addVertex(Graph* graph, int vertex) {
    if (graph->vertices == NULL) {
        graph->vertices = (int**)malloc(sizeof(int*));
        graph->vertices[0] = (int*)malloc(sizeof(int));
        *(graph->vertices[0]) = vertex;
        graph->numVertices++;
    }
    else {
        int** newVertices = (int**)realloc(graph->vertices, (graph->numVertices + 1) * sizeof(int*));
        if (newVertices == NULL) {
            printf("Memory allocation failed");
            return;
        }
        graph->vertices = newVertices;
        graph->vertices[graph->numVertices] = (int*)malloc(sizeof(int));
        *(graph->vertices[graph->numVertices]) = vertex;
        graph->numVertices++;
    }
}

void addEdge(Graph* graph, int start, int end, int weight) {
    if (graph->vertices == NULL) {
        printf("Graph is empty");
        return;
    }
    int startIdx = -1;
    int endIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == start) {
            startIdx = i;
        }
        if (*(graph->vertices[i]) == end) {
            endIdx = i;
        }
    }
    if (startIdx == -1 || endIdx == -1) {
        printf("One or both of the vertices do not exist in the graph");
        return;
    }
    int* newEdges = (int*)realloc(graph->vertices[startIdx], (2 + 2 * (*(graph->vertices[startIdx] + 1]))) * sizeof(int));
    if (newEdges == NULL) {
        printf("Memory allocation failed");
        return;
    }
    graph->vertices[startIdx] = newEdges;
    *(graph->vertices[startIdx] + 1) += 1;
    *(graph->vertices[startIdx] + 1 + 2 * (*(graph->vertices[startIdx] + 1)) - 1) = end;
    *(graph->vertices[startIdx] + 1 + 2 * (*(graph->vertices[startIdx] + 1))) = weight;
    if (!graph->directed) {
        newEdges = (int*)realloc(graph->vertices[endIdx], (2 + 2 * (*(graph->vertices[endIdx] + 1]))) * sizeof(int));
        if (newEdges == NULL) {
            printf("Memory allocation failed");
            return;
        }
        graph->vertices[endIdx] = newEdges;
        *(graph->vertices[endIdx] + 1) += 1;
        *(graph->vertices[endIdx] + 1 + 2 * (*(graph->vertices[endIdx] + 1)) - 1) = start;
        *(graph->vertices[endIdx] + 1 + 2 * (*(graph->vertices[endIdx] + 1))) = weight;
    }
}

Edge* getNeighbors(Graph* graph, int vertex, int* numNeighbors) {
    int vertexIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == vertex) {
            vertexIdx = i;
            break;
        }
    }
    if (vertexIdx == -1) {
        printf("Vertex does not exist in the graph");
        return NULL;
    }
    int numEdges = *(graph->vertices[vertexIdx] + 1);
    Edge* neighbors = (Edge*)malloc(numEdges * sizeof(Edge));
    if (neighbors == NULL) {
        printf("Memory allocation failed");
        return NULL;
    }
    for (int i = 0; i < numEdges; i++) {
        neighbors[i].end = *(graph->vertices[vertexIdx] + 1 + 2 * i);
        neighbors[i].weight = *(graph->vertices[vertexIdx] + 1 + 2 * i + 1);
    }
    *numNeighbors = numEdges;
    return neighbors;
}

void removeVertex(Graph* graph, int vertex) {
    int vertexIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == vertex) {
            vertexIdx = i;
            break;
        }
    }
    if (vertexIdx == -1) {
        printf("Vertex does not exist in the graph");
        return;
    }
    free(graph->vertices[vertexIdx]);
    for (int i = vertexIdx; i < graph->numVertices - 1; i++) {
        graph->vertices[i] = graph->vertices[i + 1];
    }
    int** newVertices = (int**)realloc(graph->vertices, (graph->numVertices - 1) * sizeof(int*));
    if (newVertices == NULL) {
        printf("Memory allocation failed");
        return;
    }
    graph->vertices = newVertices;
    graph->numVertices--;
    for (int i = 0; i < graph->numVertices; i++) {
        int numEdges = *(graph->vertices[i] + 1);
        for (int j = 0; j < numEdges; j++) {
            if (*(graph->vertices[i] + 1 + 2 * j) == vertex) {
                for (int k = j; k < numEdges - 1; k++) {
                    *(graph->vertices[i] + 1 + 2 * k) = *(graph->vertices[i] + 1 + 2 * k + 2);
                    *(graph->vertices[i] + 1 + 2 * k + 1) = *(graph->vertices[i] + 1 + 2 * k + 3);
                }
                int* newEdges = (int*)realloc(graph->vertices[i], (2 + 2 * (numEdges - 1)) * sizeof(int));
                if (newEdges == NULL) {
                    printf("Memory allocation failed");
                    return;
                }
                graph->vertices[i] = newEdges;
                *(graph->vertices[i] + 1) -= 1;
                break;
            }
        }
    }
}

void removeEdge(Graph* graph, int start, int end) {
    int startIdx = -1;
    int endIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == start) {
            startIdx = i;
        }
        if (*(graph->vertices[i]) == end) {
            endIdx = i;
        }
    }
    if (startIdx == -1 || endIdx == -1) {
        printf("One or both of the vertices do not exist in the graph");
        return;
    }
    int numEdges = *(graph->vertices[startIdx] + 1);
    for (int i = 0; i < numEdges; i++) {
        if (*(graph->vertices[startIdx] + 1 + 2 * i) == end) {
            for (int j = i; j < numEdges - 1; j++) {
                *(graph->vertices[startIdx] + 1 + 2 * j) = *(graph->vertices[startIdx] + 1 + 2 * j + 2);
                *(graph->vertices[startIdx] + 1 + 2 * j + 1) = *(graph->vertices[startIdx] + 1 + 2 * j + 3);
            }
            int* newEdges = (int*)realloc(graph->vertices[startIdx], (2 + 2 * (numEdges - 1)) * sizeof(int));
            if (newEdges == NULL) {
                printf("Memory allocation failed");
                return;
            }
            graph->vertices[startIdx] = newEdges;
            *(graph->vertices[startIdx] + 1) -= 1;
            break;
        }
    }
    if (!graph->directed) {
        numEdges = *(graph->vertices[endIdx] + 1);
        for (int i = 0; i < numEdges; i++) {
            if (*(graph->vertices[endIdx] + 1 + 2 * i) == start) {
                for (int j = i; j < numEdges - 1; j++) {
                    *(graph->vertices[endIdx] + 1 + 2 * j) = *(graph->vertices[endIdx] + 1 + 2 * j + 2);
                    *(graph->vertices[endIdx] + 1 + 2 * j + 1) = *(graph->vertices[endIdx] + 1 + 2 * j + 3);
                }
                int* newEdges = (int*)realloc(graph->vertices[endIdx], (2 + 2 * (numEdges - 1)) * sizeof(int));
                if (newEdges == NULL) {
                    printf("Memory allocation failed");
                    return;
                }
                graph->vertices[endIdx] = newEdges;
                *(graph->vertices[endIdx] + 1) -= 1;
                break;
            }
        }
    }
}

int hasVertex(Graph* graph, int vertex) {
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == vertex) {
            return 1;
        }
    }
    return 0;
}

int hasEdge(Graph* graph, int start, int end) {
    int startIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == start) {
            startIdx = i;
            break;
        }
    }
    if (startIdx == -1) {
        return 0;
    }
    int numEdges = *(graph->vertices[startIdx] + 1);
    for (int i = 0; i < numEdges; i++) {
        if (*(graph->vertices[startIdx] + 1 + 2 * i) == end) {
            return 1;
        }
    }
    return 0;
}

void removeEdges(Graph* graph, int vertex) {
    int vertexIdx = -1;
    for (int i = 0; i < graph->numVertices; i++) {
        if (*(graph->vertices[i]) == vertex) {
            vertexIdx = i;
            break;
        }
    }
    if (vertexIdx == -1) {
        printf("Vertex does not exist in the graph");
        return;
    }
    int numEdges = *(graph->vertices[vertexIdx] + 1);
    for (int i = 0; i < numEdges; i++) {
        int end = *(graph->vertices[vertexIdx] + 1 + 2 * i);
        int endIdx = -1;
        for (int j = 0; j < graph->numVertices; j++) {
            if (*(graph->vertices[j]) == end) {
                endIdx = j;
                break;
            }
        }
        if (endIdx == -1) {
            printf("Vertex does not exist in the graph");
            return;
        }
        for (int j = 0; j < *(graph->vertices[endIdx] + 1); j++) {
            if (*(graph->vertices[endIdx] + 1 + 2 * j) == vertex) {
                for (int k = j; k < *(graph->vertices[endIdx] + 1) - 1; k++) {
                    *(graph->vertices[endIdx] + 1 + 2 * k) = *(graph->vertices[endIdx] + 1 + 2 * k + 2);
                    *(graph->vertices[endIdx] + 1 + 2 * k + 1) = *(graph->vertices[endIdx] + 1 + 2 * k + 3);
                }
                int* newEdges = (int*)realloc(graph->vertices[endIdx], (2 + 2 * (*(graph->vertices[endIdx] + 1)) - 2) * sizeof(int));
                if (newEdges == NULL) {
                    printf("Memory allocation failed");
                    return;
                }
                graph->vertices[endIdx] = newEdges;
                *(graph->vertices[endIdx] + 1) -= 1;
                break;
            }
        }
    }
    free(graph->vertices[vertexIdx]);
    for (int i = vertexIdx; i < graph->numVertices - 1; i++) {
        graph->vertices[i] = graph->vertices[i + 1];
    }
    int** newVertices = (int**)realloc(graph->vertices, (graph->numVertices - 1) * sizeof(int*));
    if (newVertices == NULL) {
        printf("Memory allocation failed");
        return;
    }
    graph->vertices = newVertices;
    graph->numVertices--;
}

int numVertices(Graph* graph) {
    return graph->numVertices;
}

int numEdges(Graph* graph) {
    int edgeCount = 0;
    for (int i = 0; i < graph->numVertices; i++) {
        edgeCount += *(graph->vertices[i] + 1);
    }
    return graph->directed ? edgeCount : edgeCount / 2;
}

void dfs(Graph* graph, int start, int* visited, int* numVisited) {
    visited[start] = 1;
    (*numVisited)++;
    int numNeighbors = 0;
    Edge* neighbors = getNeighbors(graph, start, &numNeighbors);
    for (int i = 0; i < numNeighbors; i++) {
        if (!visited[neighbors[i].end]) {
            dfs(graph, neighbors[i].end, visited, numVisited);
        }
    }
    free(neighbors);
}

int* depthFirstSearch(Graph* graph, int start, int* numVisited) {
    int* visited = (int*)calloc(graph->numVertices, sizeof(int));
    if (visited == NULL) {
        printf("Memory allocation failed");
        return NULL;
    }
    *numVisited = 0;
    dfs(graph, start, visited, numVisited);
    return visited;
}

int* breadthFirstSearch(Graph* graph, int start, int* numVisited) {
    int* visited = (int*)calloc(graph->numVertices, sizeof(int));
    if (visited == NULL) {
        printf("Memory allocation failed");
        return NULL;
    }
    *numVisited = 0;
    int* queue = (int*)malloc(graph->numVertices * sizeof(int));
    if (queue == NULL) {
        printf("Memory allocation failed");
        return NULL;
    }
    int front = 0;
    int rear = 0;
    queue[rear++] = start;
    visited[start] = 1;
    (*numVisited)++;
    while (front < rear) {
        int vertex = queue[front++];
        int numNeighbors = 0;
        Edge* neighbors = getNeighbors(graph, vertex, &numNeighbors);
        for (int i = 0; i < numNeighbors; i++) {
            if (!visited[neighbors[i].end]) {
                queue[rear++] = neighbors[i].end;
                visited[neighbors[i].end] = 1;
                (*numVisited)++;
            }
        }
        free(neighbors);
    }
    free(queue);
    return visited;
}

double shortestPathDijkstra(Graph* graph, int start, int end) {
    double* distances = (double*)malloc(graph->numVertices * sizeof(double));
    if (distances == NULL) {
        printf("Memory allocation failed");
        return -1;
    }
    for (int i = 0; i < graph->numVertices; i++) {
        distances[i] = INFINITY;
    }
    distances[start] = 0;
    int* visited = (int*)calloc(graph->numVertices, sizeof(int));
    if (visited == NULL) {
        printf("Memory allocation failed");
        return -1;
    }
    int numVisited = 0;
    while (numVisited < graph->numVertices) {
        int minDistanceVertex = -1;
        double minDistance = INFINITY;
        for (int i = 0; i < graph->numVertices; i++) {
            if (!visited[i] && distances[i] < minDistance) {
                minDistanceVertex = i;
                minDistance = distances[i];
            }
        }
        if (minDistanceVertex == -1) {
            break;
        }
        visited[minDistanceVertex] = 1;
        numVisited++;
        int numNeighbors = 0;
        Edge* neighbors = getNeighbors(graph, minDistanceVertex, &numNeighbors);
        for (int i = 0; i < numNeighbors; i++) {
            double distance = distances[minDistanceVertex] + neighbors[i].weight;
            if (distance < distances[neighbors[i].end]) {
                distances[neighbors[i].end] = distance;
            }
        }
        free(neighbors);
    }
    double shortestPath = distances[end];
    free(distances);
    free(visited);
    return shortestPath;
}

int isConnected(Graph* graph) {
    if (graph->numVertices == 0) {
        printf("Graph is empty");
        return 0;
    }
    int* visited = depthFirstSearch(graph, *(graph->vertices[0]), &(int){0});
    int connected = *numVisited == graph->numVertices;
    free(visited);
    return connected;
}

int isSortedDescending(int* A, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (A[i] < A[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int isSorted(int* A, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (A[i] > A[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int hasDuplicates(int* nums, int n) {
    int seen = 0;
    for (int i = 0; i < n; i++) {
        if ((seen & (1 << nums[i])) > 0) {
            return 1;
        }
        seen |= 1 << nums[i];
    }
    return 0;
}

char* fileHash(char* filePath, char* hashAlgorithm) {
    if (strcmp(hashAlgorithm, "sha256") != 0) {
        printf("Invalid hash algorithm");
        return NULL;
    }
    FILE* file = fopen(filePath, "rb");
    if (file == NULL) {
        printf("Failed to open file");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);
    char* buffer = (char*)malloc(fileSize);
    if (buffer == NULL) {
        printf("Memory allocation failed");
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, fileSize, file) != fileSize) {
        printf("Failed to read file");
        fclose(file);
        free(buffer);
        return NULL;
    }
    fclose(file);
    unsigned char hash[32];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, buffer, fileSize);
    SHA256_Final(hash, &sha256);
    free(buffer);
    char* hashString = (char*)malloc(65);
    if (hashString == NULL) {
        printf("Memory allocation failed");
        return NULL;
    }
    for (int i = 0; i < 32; i++) {
        sprintf(hashString + (i * 2), "%02x", hash[i]);
    }
    hashString[64] = '\0';
    return hashString;
}

int hasDuplicateValues(int* lst, int n) {
    int seen = 0;
    for (int i = 0; i < n; i++) {
        int itemMask = 1 << lst[i];
        if ((seen & itemMask) != 0) {
            return 1;
        }
        seen |= itemMask;
    }
    return 0;
}

char* calculateLn2(int precision) {
    mpf_set_default_prec(precision + 2);
    mpf_t ln2;
    mpf_init(ln2);
    mpf_set_d(ln2, 0);
    for (int n = 1; n < precision * 2; n++) {
        mpf_t term;
        mpf_init(term);
        mpf_set_d(term, 1);
        mpf_div_ui(term, term, n);
        mpf_pow_ui(term, term, n - 1);
        if (n % 2 == 0) {
            mpf_neg(term, term);
        }
        mpf_add(ln2, ln2, term);
        mpf_clear(term);
    }
    char* ln2String = (char*)malloc(precision + 1);
    if (ln2String == NULL) {
        printf("Memory allocation failed");
        mpf_clear(ln2);
        return NULL;
    }
    mp_exp_t exp;
    mpf_get_str(ln2String, &exp, 10, precision, ln2);
    ln2String[precision] = '\0';
    mpf_clear(ln2);
    return ln2String;
}

char* calculatePi(int precision) {
    mpf_set_default_prec(precision + 2);
    mpf_t pi;
    mpf_init(pi);
    mpf_set_d(pi, 0);
    mpf_t term;
    mpf_init(term);
    mpf_t one;
    mpf_init(one);
    mpf_set_d(one, 1);
    mpf_t five;
    mpf_init(five);
    mpf_set_d(five, 5);
    mpf_t twoHundredThirtyNine;
    mpf_init(twoHundredThirtyNine);
    mpf_set_d(twoHundredThirtyNine, 239);
    for (int k = 0; k < precision; k++) {
        mpf_t numerator1;
        mpf_init(numerator1);
        mpf_set_d(numerator1, 4);
        mpf_pow_ui(numerator1, numerator1, k);
        mpf_t denominator1;
        mpf_init(denominator1);
        mpf_set_d(denominator1, 8 * k + 1);
        mpf_t denominator2;
        mpf_init(denominator2);
        mpf_set_d(denominator2, 8 * k + 4);
        mpf_t denominator3;
        mpf_init(denominator3);
        mpf_set_d(denominator3, 8 * k + 5);
        mpf_t denominator4;
        mpf_init(denominator4);
        mpf_set_d(denominator4, 8 * k + 6);
        mpf_t term1;
        mpf_init(term1);
        mpf_div(term1, numerator1, denominator1);
        mpf_t term2;
        mpf_init(term2);
        mpf_div(term2, one, denominator2);
        mpf_t term3;
        mpf_init(term3);
        mpf_div(term3, one, denominator3);
        mpf_t term4;
        mpf_init(term4);
        mpf_div(term4, one, denominator4);
        mpf_t term5;
        mpf_init(term5);
        mpf_sub(term5, term2, term3);
        mpf_sub(term5, term5, term4);
        mpf_mul(term5, term5, term1);
        mpf_add(pi, pi, term5);
        mpf_clear(numerator1);
        mpf_clear(denominator1);
        mpf_clear(denominator2);
        mpf_clear(denominator3);
        mpf_clear(denominator4);
        mpf_clear(term1);
        mpf_clear(term2);
        mpf_clear(term3);
        mpf_clear(term4);
        mpf_clear(term5);
    }
    mpf_t term6;
    mpf_init(term6);
    mpf_div(term6, one, five);
    for (int k = 0; k < precision; k++) {
        mpf_t numerator2;
        mpf_init(numerator2);
        mpf_set_d(numerator2, 1);
        mpf_pow_ui(numerator2, numerator2, k);
        mpf_t denominator5;
        mpf_init(denominator5);
        mpf_set_d(denominator5, 8 * k + 1);
        mpf_t denominator6;
        mpf_init(denominator6);
        mpf_set_d(denominator6, 8 * k + 4);
        mpf_t denominator7;
        mpf_init(denominator7);
        mpf_set_d(denominator7, 8 * k + 5);
        mpf_t denominator8;
        mpf_init(denominator8);
        mpf_set_d(denominator8, 8 * k + 6);
        mpf_t term7;
        mpf_init(term7);
        mpf_div(term7, numerator2, denominator5);
        mpf_t term8;
        mpf_init(term8);
        mpf_div(term8, one, denominator6);
        mpf_t term9;
        mpf_init(term9);
        mpf_div(term9, one, denominator7);
        mpf_t term10;
        mpf_init(term10);
        mpf_div(term10, one, denominator8);
        mpf_t term11;
        mpf_init(term11);
        mpf_sub(term11, term8, term9);
        mpf_sub(term11, term11, term10);
        mpf_mul(term11, term11, term7);
        mpf_div(term11, term11, twoHundredThirtyNine);
        mpf_sub(term6, term6, term11);
        mpf_clear(numerator2);
        mpf_clear(denominator5);
        mpf_clear(denominator6);
        mpf_clear(denominator7);
        mpf_clear(denominator8);
        mpf_clear(term7);
        mpf_clear(term8);
        mpf_clear(term9);
        mpf_clear(term10);
        mpf_clear(term11);
    }
    mpf_add(pi, pi, term6);
    char* piString = (char*)malloc(precision + 1);
    if (piString == NULL) {
        printf("Memory allocation failed");
        mpf_clear(pi);
        mpf_clear(term);
        mpf_clear(one);
        mpf_clear(five);
        mpf_clear(twoHundredThirtyNine);
        return NULL;
    }
    mp_exp_t exp;
    mpf_get_str(piString, &exp, 10, precision, pi);
    piString[precision] = '\0';
    mpf_clear(pi);
    mpf_clear(term);
    mpf_clear(one);
    mpf_clear(five);
    mpf_clear(twoHundredThirtyNine);
    return piString;
}

char* calculateEWithPrecision(int precision) {
    mpf_set_default_prec(precision + 2);
    mpf_t e;
    mpf_init(e);
    mpf_set_d(e, 1);
    mpf_t factorial;
    mpf_init(factorial);
    mpf_set_d(factorial, 1);
    for (int i = 1; i < precision * 2; i++) {
        mpf_mul_ui(factorial, factorial, i);
        mpf_t term;
        mpf_init(term);
        mpf_div(term, one, factorial);
        mpf_add(e, e, term);
        mpf_clear(term);
    }
    char* eString = (char*)malloc(precision + 1);
    if (eString == NULL) {
        printf("Memory allocation failed");
        mpf_clear(e);
        mpf_clear(factorial);
        return NULL;
    }
    mp_exp_t exp;
    mpf_get_str(eString, &exp, 10, precision, e);
    eString[precision] = '\0';
    mpf_clear(e);
    mpf_clear(factorial);
    return eString;
}

char* goldenRatio(int precision) {
    mpf_set_default_prec(precision + 2);
    mpf_t phi;
    mpf_init(phi);
    mpf_t numerator;
    mpf_init(numerator);
    mpf_set_d(numerator, 1);
    mpf_t denominator;
    mpf_init(denominator);
    mpf_set_d(denominator, 1);
    mpf_add(phi, numerator, denominator);
    for (int i = 0; i < precision; i++) {
        mpf_swap(numerator, denominator);
        mpf_add(denominator, numerator, denominator);
        mpf_swap(phi, denominator);
    }
    char* phiString = (char*)malloc(precision + 1);
    if (phiString == NULL) {
        printf("Memory allocation failed");
        mpf_clear(phi);
        mpf_clear(numerator);
        mpf_clear(denominator);
        return NULL;
    }
    mp_exp_t exp;
    mpf_get_str(phiString, &exp, 10, precision, phi);
    phiString[precision] = '\0';
    mpf_clear(phi);
    mpf_clear(numerator);
    mpf_clear(denominator);
    return phiString;
}

double seidelConstant(int** A, int n, double tol, int maxIterations) {
    double* x = (double*)calloc(n, sizeof(double));
    if (x == NULL) {
        printf("Memory allocation failed");
        return -1;
    }
    double* xNew = (double*)calloc(n, sizeof(double));
    if (xNew == NULL) {
        printf("Memory allocation failed");
        free(x);
        return -1;
    }
    for (int i = 0; i < maxIterations; i++) {
        double errorNorm = 0;
        for (int j = 0; j < n; j++) {
            xNew[j] = A[j][n];
            for (int k = 0; k < n; k++) {
                if (k != j) {
                    xNew[j] -= A[j][k] * xNew[k];
                }
            }
            xNew[j] /= A[j][j];
            errorNorm += fabs(xNew[j] - x[j]);
        }
        if (errorNorm < tol) {
            free(x);
            double maxEigenvalueL = 0;
            double maxEigenvalueU = 0;
            for (int j = 0; j < n; j++) {
                double sumL = 0;
                double sumU = 0;
                for (int k = 0; k < j; k++) {
                    sumL += A[j][k] * xNew[k];
                }
                for (int k = j + 1; k < n; k++) {
                    sumU += A[j][k] * xNew[k];
                }
                double eigenvalueL = (A[j][j] * xNew[j] - sumL) / xNew[j];
                double eigenvalueU = (A[j][j] * xNew[j] - sumU) / xNew[j];
                if (eigenvalueL > maxEigenvalueL) {
                    maxEigenvalueL = eigenvalueL;
                }
                if (eigenvalueU > maxEigenvalueU) {
                    maxEigenvalueU = eigenvalueU;
                }
            }
            return 1 / (1 - maxEigenvalueL / maxEigenvalueU);
        }
        memcpy(x, xNew, n * sizeof(double));
    }
    free(x);
    free(xNew);
    printf("Seidel constant did not converge within the specified number of iterations");
    return -1;
}

int log(int x, int base) {
    if (x <= 0 || base <= 0 || base == 1) {
        printf("The number and base must be positive, and the base must not be 1");
        return -1;
    }
    int result = 0;
    while (x >= base) {
        x /= base;
        result++;
    }
    return result;
}

double tetration(double base, int n) {
    double result = 1;
    while (n > 0) {
        result = pow(base, result);
        n--;
    }
    return result;
}

int main() {
    // Test code
    return 0;
}
