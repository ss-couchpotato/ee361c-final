#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	node() {
		reset();
	}

	void reset() {
		cost = INT_MAX;
		parent = -1;
		visited = false;
	}
};

// Globals
int blockSize;

void printResults(vector<node> graph)
{
	cout << "Vertex\t Distance from start_node\n";
	for (int i = 0; i < graph.size(); i++) {
		cout << i << "\t" << graph[i].cost << "\n";
	}
}

void compareResults(vector<node> graph1, vector<node> graph2)
{
	cout << "****** RESULT ******\n";
	for (int i = 0; i < graph1.size(); i++) {
		if (graph1[i].cost != graph2[i].cost) {
			cout << "INCORRECT: Parallel and Serial SPA do NOT match\n";
			return;
		}
	}
	cout << "SUCCESS: Parallel and Serial SPA match\n";
}

// helper function that finds nearest neighbor for serial SPA
int minDistance(vector<node> &graph, list<int> &queue)
{
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
vector<node> SPA_serial(vector<node> &graph, int start_node)
{
	int num_nodes = graph.size();
	list<int> queue;

	for (int i = 0; i < num_nodes; i++) {
		graph[i].reset();
		queue.push_back(i);
	}
	graph[start_node].cost = 0;

	// find shortest path from start_node to each node
	while (!queue.empty()) {
    for (int i = 0; i < num_nodes; i++) {
      printf("cost for finding min %d: %d\n", i, graph[i].cost);
    }
		int nearest_node = minDistance(graph, queue);	// Get node with lowest cost from queue
    printf("nearest node is %d\n", nearest_node);
		queue.remove(nearest_node);
		graph[nearest_node].visited = true;
		vector<edge> &neighbors = graph[nearest_node].neighbors;

		for (edge const& neighbor : neighbors) {
			int weight = neighbor.weight;
			int curr_node = neighbor.neighbor;
			if (graph[curr_node].visited)
				continue;

			int cost = graph[nearest_node].cost + weight;
			if (cost < graph[curr_node].cost) {
				graph[curr_node].cost = cost;
				graph[curr_node].parent = nearest_node;
			}
		}

    for (int i = 0; i < num_nodes; i++) {
      printf("cost %d: %d\n", i, graph[i].cost);
    }
	}
	return graph;
}

// Find cuda enabled device and return block size
static int getBlockSize()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess || deviceCount == 0)
		PFERROR_EXIT(1, "Error locating CUDA-enabled device");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.maxThreadsPerBlock;
}

__global__ void checkMin(int n, int *input, bool *is_min, bool *visited)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n * n && (input[index / n] > input[index % n] || visited))
		is_min[index / n] = false;
}

__global__ void findMinIdx(int n, bool *is_min, int *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n && is_min[index]) {
		*result = index;
	}
}

int parallel_min_distance(bool *visited, int *cost, int n) {
	int  *c_result, result;
	bool *c_is_min;
	cudaMalloc((void **)&c_is_min, sizeof(bool) * n);
	cudaMalloc((void **)&c_result, sizeof(int));
	cudaMemset(c_is_min, true, sizeof(bool) * n);
	int numBlocks = (n * n + blockSize - 1) / blockSize;
	checkMin << <numBlocks, blockSize >> > (n, cost, c_is_min, visited);
	cudaDeviceSynchronize();
	numBlocks = (n + blockSize - 1) / blockSize;
	findMinIdx << <numBlocks, blockSize >> > (n, c_is_min, c_result);

	// Retrieving result
	cudaMemcpy((void *)&result, (void *)c_result, sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}

__global__ void update_cost(int *matrix, bool *visited, int *costs, int n, int node) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n && index != node) {
    if (visited[index] || matrix[node*n+index] == INT_MAX)
      return;
    int cost = costs[node] + matrix[node*n+index];
    if (cost < costs[index]) {
      costs[index] = cost;
    }
  }
}

vector<node> SPA_parallel(int *matrix, int n, int start_node) {
  int *c_matrix, *c_cost;
  bool *c_visited;
  int num_node = n;
  int *cost = (int *) malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) {
    cost[i] = INT_MAX;
  }
  cudaMalloc((void **)&c_matrix, sizeof(int) * n * n);
  cudaMalloc((void **)&c_cost, sizeof(int) * n);
  cudaMalloc((void **)&c_visited, sizeof(bool) * n);
  cudaMemcpy(c_matrix, matrix, sizeof(int) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(c_cost, cost, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemset((void *)&c_cost[start_node], 0, sizeof(int));
  int numBlocks = (n + blockSize - 1) / blockSize;

	// find shortest path from start_node to each node
	while (num_node != 0) {
    // In parallel, get node with lowest cost from queue
		int nearest_node = parallel_min_distance(c_visited, c_cost, n);
    printf("nearest node is %d\n", nearest_node);
	  cudaMemset((void *)&c_visited[nearest_node], true, sizeof(bool));

    // In parallel, update all neighbors of nearest node
    update_cost<<<numBlocks, blockSize>>>(matrix, c_visited, c_cost, n, nearest_node);
    cudaDeviceSynchronize();
    cudaMemcpy(cost, c_cost, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
      printf("cost %d: %d\n", i, cost[i]);
    }
    num_node--;
	}
  cudaMemcpy(cost, c_cost, sizeof(int) * n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    printf("%d: %d\n", i, cost[i]);
  vector<node> graph(n);
  for (int i = 0; i < n; i++) {
    graph[i].cost = cost[i];
  }
  cudaFree(c_matrix);
  cudaFree(c_cost);
  cudaFree(c_visited);
	return graph;
}

int main(int argc, char* argv[])
{
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
  int *matrix = (int *) malloc(sizeof(int) * num_node * num_node);
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
    matrix[a*num_node+b] += weight;
    matrix[b*num_node+a] += weight;
	}
  for (int i = 0; i < num_node; i++) {
    for (int j = 0; j < num_node; j++) {
      if (matrix[i*num_node+j] != 0)
        graph[i].neighbors.push_back(edge(matrix[i*num_node+j], j, i));
    }
  }
	clock_t start = clock();
	vector<node> graphParallel = SPA_parallel(matrix, num_node, 0);
	clock_t end = clock();
	cout << "Parallel shortest distance algorithm took " << (end - start) << " clock cycles\n\n";

	start = clock();
	vector<node> graphSerial = SPA_serial(graph, 0); // find shortest distance from node 0 to all others
	end = clock();
	cout << "Serial shortest distance algorithm took " << (end - start) << " clock cycles\n\n";

  printResults(graphSerial);
  printResults(graphParallel);
	compareResults(graphParallel, graphSerial);
	return 0;
}
