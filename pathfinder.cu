#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <sstream>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <thrust/extrema.h>

using namespace std;

#define PERROR_EXIT(message)                                                   \
  do {                                                                         \
    perror((message));                                                         \
    exit(4);                                                                   \
  } while (0)

#define PFERROR_EXIT(exit_code, ...)                                           \
  do {                                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
    exit(exit_code);                                                           \
  } while (0)

class edge {
public:
  int parent;
  int weight;
  int neighbor;
  edge() {
    this->weight = 0;
    this->neighbor = 0;
    this->parent = 0;
  }
  edge(int weight, int neighbor, int parent) {
    this->weight = weight;
    this->neighbor = neighbor;
    this->parent = parent;
  }
};

class node {
public:
  int cost;
  int parent;
  bool visited;
  vector<edge> neighbors;

  node() { reset(); }

  void reset() {
    cost = INT_MAX;
    parent = -1;
    visited = false;
  }
};

// Globals
int blockSize;

void printResults(vector<node> graph) {
  cout << "Vertex\t Distance from start_node\n";
  for (int i = 0; i < graph.size(); i++) {
    cout << i << "\t" << graph[i].cost << "\n";
  }
}

void compareResults(vector<node> graph1, vector<node> graph2) {
  cout << "****** RESULT ******\n";
  bool success = true;
  for (int i = 0; i < graph1.size(); i++) {
    if (graph1[i].cost != graph2[i].cost) {
      printf("cost 1: %d\tcost 2: %d\n", graph1[i].cost, graph2[i].cost);
      success = false;
      ;
    }
  }
  if (success)
    cout << "SUCCESS: Parallel and Serial SPA match\n";
  else
    cout << "INCORRECT: Parallel and Serial SPA do NOT match\n";
}

// helper function that finds nearest neighbor for serial SPA
int minDistance(vector<node> &graph, list<int> &queue) {
  int min_node = queue.front();
  int min_distance = graph[min_node].cost;

  for (int node : queue) {
    if (graph[node].cost < min_distance) {
      min_distance = graph[node].cost, min_node = node;
    }
  }
  return min_node;
}

// serial implementation of Dijkstra's shortest path algorithm
vector<node> SPA_serial(vector<node> &graph, int start_node) {
  int num_nodes = graph.size();
  list<int> queue;

  for (int i = 0; i < num_nodes; i++) {
    graph[i].reset();
    queue.push_back(i);
  }
  graph[start_node].cost = 0;

  // find shortest path from start_node to each node
  while (!queue.empty()) {
    int nearest_node =
        minDistance(graph, queue); // Get node with lowest cost from queue
    queue.remove(nearest_node);
    graph[nearest_node].visited = true;
    vector<edge> &neighbors = graph[nearest_node].neighbors;

    for (edge const &neighbor : neighbors) {
      int weight = neighbor.weight;
      int curr_node = neighbor.neighbor;
      if (graph[curr_node].visited || graph[nearest_node].cost == INT_MAX)
        continue;

      int cost = graph[nearest_node].cost + weight;
      if (cost < graph[curr_node].cost) {
        graph[curr_node].cost = cost;
        graph[curr_node].parent = nearest_node;
      }
    }
  }
  return graph;
}

// Find cuda enabled device and return block size
static int getBlockSize() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess || deviceCount == 0)
    PFERROR_EXIT(1, "Error locating CUDA-enabled device");
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return deviceProp.maxThreadsPerBlock;
}

__global__ void checkMin(int n, int *input, bool *is_min, bool *visited) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n * n &&
      (input[index / n] > input[index % n] || visited[index / n])) {
    // If other node already visited, don't count it
    if (visited[index % n])
      return;
    is_min[index / n] = false;
  }
}

__global__ void findMinIdx(int n, bool *is_min, int *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n && is_min[index]) {
    *result = index;
  }
}

// is_min, visited, cost, must be device ptrs
int parallel_min_distance(bool *is_min, bool *visited, int *cost, int n) {
  int *c_result, result;
  cudaMalloc((void **)&c_result, sizeof(int));
  cudaMemset(is_min, true, sizeof(bool) * n);
  int numBlocks = n * n / blockSize + 1;
  checkMin<<<numBlocks, blockSize>>>(n, cost, is_min, visited);
  cudaDeviceSynchronize();
  numBlocks = n / blockSize + 1;
  findMinIdx<<<numBlocks, blockSize>>>(n, is_min, c_result);
  cudaDeviceSynchronize();
  // Retrieving result
  cudaMemcpy((void *)&result, (void *)c_result, sizeof(int),
             cudaMemcpyDeviceToHost);
  return result;
}

__global__ void convertCost(bool *visited, int *cost, int *buffer, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    buffer[index] = visited[index] ? INT_MAX : cost[index];
  }
}

__global__ void convertCostToFloat(bool *visited, int *cost, float *buffer, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    buffer[index] = (visited[index] ? INT_MAX : cost[index]) * 1.0;
  }
}

int cublas_min_distance(float *buffer, bool *visited, int* cost, int n)
{
    int result;
    cublasHandle_t handle;
    cublasCreate(&handle);
    int numBlocks = n / blockSize + 1;
    convertCostToFloat<<<numBlocks, blockSize>>>(visited, cost, buffer, n);

    if (cublasIsamin(handle, n, buffer, 1, &result) != CUBLAS_STATUS_SUCCESS)
        printf("min failed\n");

    cublasDestroy(handle);
    return result-1;
}

int thrust_min_distance(int* cost, int n)
{
    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(cost);
    return thrust::min_element(ptr, ptr + n) - ptr;
}

__global__ void update_cost(int *matrix, bool *visited, int *costs, int *min_costs, int n,
                            int node) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  visited[node] = true;
  min_costs[node] = INT_MAX;

  if (index < n && index != node) {
    if (visited[index] || matrix[node * n + index] == INT_MAX ||
        costs[node] == INT_MAX)
      return;
    int cost = costs[node] + matrix[node * n + index];
    if (cost < costs[index]) {
      // printf("relaxing %d cost to %d from node %d using edge w/ weight %d\n",
      //   index, cost, node, matrix[node*n+index]);
      costs[index] = cost;
      min_costs[index] = cost;
    }
  }
}

vector<node> SPA_parallel(int *matrix, int n, int start_node) {
  int *c_matrix, *c_cost, *c_min_cost;
  bool *c_visited;
  // bool *c_is_min;
  // float *buffer;
  // int *buffer;
  int num_node = n;
  int *cost = (int *)malloc(sizeof(int) * n);
  if (cost == NULL)
    PERROR_EXIT("malloc");
  for (int i = 0; i < n; i++) {
    cost[i] = INT_MAX;
  }
  cost[start_node] = 0;
  cudaMalloc((void **)&c_matrix, sizeof(int) * n * n);
  cudaMalloc((void **)&c_cost, sizeof(int) * n);
  cudaMalloc((void **)&c_min_cost, sizeof(int) * n);
  cudaMalloc((void **)&c_visited, sizeof(bool) * n);
  cudaMemcpy(c_matrix, matrix, sizeof(int) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(c_cost, cost, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(c_min_cost, c_cost, sizeof(int) * n, cudaMemcpyDeviceToDevice);
  cudaMemset((void *)c_visited, false, sizeof(bool) * n);

  // cudaMalloc((void **)&c_is_min, sizeof(bool) * n);
  // cudaMalloc((void **)&buffer, sizeof(float) * n);
  // cudaMalloc((void **)&buffer, sizeof(int) * n);

  int numBlocks = (n + blockSize - 1) / blockSize;

  // find shortest path from start_node to each node
  while (num_node != 0) {
    // In parallel, get node with lowest cost from queue
    // int nearest_node = parallel_min_distance(c_is_min, c_visited, c_cost, n);
    // int nearest_node = cublas_min_distance(buffer, c_visited, c_cost, n);
    int nearest_node = thrust_min_distance(c_min_cost, n);
    // In parallel, update all neighbors of nearest node
    update_cost<<<numBlocks, blockSize>>>(c_matrix, c_visited, c_cost, c_min_cost, n,
                                          nearest_node);
    cudaDeviceSynchronize();
    num_node--;
  }
  cudaMemcpy(cost, c_cost, sizeof(int) * n, cudaMemcpyDeviceToHost);
  vector<node> graph(n);
  for (int i = 0; i < n; i++) {
    graph[i].cost = cost[i];
  }
  cudaFree(c_matrix);
  cudaFree(c_cost);
  cudaFree(c_visited);

  cudaFree(c_min_cost);
  // cudaFree(c_is_min);
  // cudaFree(buffer);
  return graph;
}

int main(int argc, char *argv[]) {
  if (argc < 2)
    PFERROR_EXIT(1, "No input file\n");

  ifstream infile(argv[1]);
  if (!infile)
    PFERROR_EXIT(1, "Error opening input file\n");

  blockSize = getBlockSize();

  string token;
  getline(infile, token, ' ');
  int num_node = stoi(token);
  getline(infile, token);
  int num_edge = stoi(token);
  // int matrix[num_node][num_node];
  int *matrix = (int *)malloc(sizeof(int) * num_node * num_node);
  if (matrix == NULL)
    PERROR_EXIT("malloc");
  memset(matrix, 0, sizeof(int) * num_node * num_node);
  vector<node> graph(num_node);
  vector<edge> edges(2 * num_edge);
  printf("n = %d\tm = %d\n", num_node, num_edge);
  for (int i = 0; i < num_edge; i++) {
    getline(infile, token, ' ');
    int a = stoi(token);
    getline(infile, token, ' ');
    int b = stoi(token);
    getline(infile, token);
    int weight = stoi(token);
    matrix[a * num_node + b] += weight;
    matrix[b * num_node + a] += weight;
  }
  for (int i = 0; i < num_node; i++) {
    for (int j = 0; j < num_node; j++) {
      if (matrix[i * num_node + j] != 0)
        graph[i].neighbors.push_back(edge(matrix[i * num_node + j], j, i));
      else
        matrix[i * num_node + j] = INT_MAX;
    }
  }
  printf("graph generated\n");
  clock_t start = clock();
  vector<node> graphParallel = SPA_parallel(matrix, num_node, 0);
  clock_t end = clock();
  cout << "Parallel shortest distance algorithm took " << (end - start)
       << " clock cycles\n\n";

  start = clock();
  vector<node> graphSerial =
      SPA_serial(graph, 0); // find shortest distance from node 0 to all others
  end = clock();
  cout << "Serial shortest distance algorithm took " << (end - start)
       << " clock cycles\n\n";

  // printResults(graphSerial);
  // printResults(graphParallel);
  compareResults(graphParallel, graphSerial);
  return 0;
}
