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

	for (list<int>::iterator it = queue.begin(); it != queue.end(); ++it){
		int node = *it;
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
		int nearest_node = minDistance(graph, queue);	// Get node with lowest cost from queue
		queue.remove(nearest_node);
		graph[nearest_node].visited = true;
		vector<edge> &neighbors = graph[nearest_node].neighbors;
		for (vector<edge>::iterator it = neighbors.begin(); it != neighbors.end(); ++it){
			edge &neighbor = *it;
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
	}
	printResults(graph);
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

__global__ void checkMin(int n, int *input, bool *is_min)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n * n && input[index / n] > input[index % n])
		is_min[index / n] = false;
}

__global__ void findMinIdx(int n, bool *is_min, int *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n && is_min[index]) {
		*result = index;
	}
}

int min_val(int n, int *c_input) {
	int  *c_result, result;
	bool *c_is_min;
	//cudaMalloc((void **)&c_input, sizeof(int) * n);
	cudaMalloc((void **)&c_is_min, sizeof(bool) * n);
	cudaMalloc((void **)&c_result, sizeof(int));
	//cudaMemcpy(c_input, input, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemset(c_is_min, true, sizeof(bool) * n);
	int blockSize = getBlockSize();
	int numBlocks = (n * n + blockSize - 1) / blockSize;
	checkMin << <numBlocks, blockSize >> > (n, c_input, c_is_min);
	cudaDeviceSynchronize();
	numBlocks = (n + blockSize - 1) / blockSize;
	findMinIdx << <numBlocks, blockSize >> > (n, c_is_min, c_result);

	// Retrieving result
	cudaMemcpy((void *)&result, (void *)c_result, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("WOW index:%d is min", result);
	return result;
}

__global__ void compute_edge_dist(int num_edges, int * cost, node * graph, edge * edges) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num_edges) {
		int temp_cost = INT_MAX;
		edge *e = &edges[index];
		if (graph[e->parent].visited && !graph[e->neighbor].visited)
			temp_cost = graph[e->parent].cost + e->weight;
		//cost of host (from graph) + edge weight
		cost[index] = (temp_cost >= 0) ? temp_cost : INT_MAX;
		//printf("index:%d is %d\n", index, cost[index]);
	}
}

__global__ void update_costs(int edge_index, int num_to_update, int * cost, node * graph, edge * edges, edge * edges_to_update) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int cost_of_added = cost[edge_index];
	if (index < num_to_update) {
		int new_cost = edges_to_update[index].weight + cost_of_added;
		if (new_cost >= 0 && new_cost < graph[edges_to_update[index].neighbor].cost)graph[edges_to_update[index].neighbor].cost = new_cost;
	}
	else if (index == num_to_update) {
		graph[edges[edge_index].neighbor].visited = true;
		graph[edges[edge_index].neighbor].cost = cost_of_added;
	}
}

__global__ void update_costs_init(int num_to_update, int cost_of_added, node * graph, edge * edges_to_update) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num_to_update) {
		int new_cost = edges_to_update[index].weight + cost_of_added;
		if (new_cost < graph[edges_to_update[index].neighbor].cost)graph[edges_to_update[index].neighbor].cost = new_cost;
	}
}

void init_start_edges(vector<node> &graph, node * device_graph, int start_node) {
	int number_of_edges_to_update = graph[start_node].neighbors.size();
	//get raw data
	edge * edges_to_update = &(graph[start_node].neighbors[0]);
	edge* device_edges_to_update;
	cudaMalloc((void **)&device_edges_to_update, sizeof(edge)*number_of_edges_to_update);
	cudaMemcpy(device_edges_to_update, edges_to_update, sizeof(edge)*number_of_edges_to_update, cudaMemcpyHostToDevice);
	int blockSize = getBlockSize();
	int numBlocks = ((number_of_edges_to_update)+blockSize - 1) / blockSize;
	update_costs_init << <numBlocks, blockSize >> > (number_of_edges_to_update, 0, device_graph, device_edges_to_update);
	cudaFree(device_edges_to_update);
}

vector<node> SPA_parallel(vector<node> &graph, vector<edge> &edges, int start_node) {
	int num_edges = edges.size();
	int num_nodes = graph.size();
	int num_added = 0;
	for (int i = 0; i < num_nodes; i++) {
		graph[i].reset();
	}
	graph[start_node].visited = true;
	graph[start_node].cost = 0;
	//Copying edege and graph data to Device
	node* host_graph = &graph[0];
	node* device_graph;
	cudaMalloc((void **)&device_graph, sizeof(node)*num_nodes);
	cudaMemcpy(device_graph, host_graph, sizeof(node)*num_nodes, cudaMemcpyHostToDevice);
	//set all neighbors of start node to their respective weights in parallel
	init_start_edges(graph, device_graph, start_node);
	/*
	for (edge const& e : graph[start_node].neighbors) {
		graph[e.neighbor].cost = e.weight;
	}*/

	edge* host_edge = &edges[0];
	edge* device_edge;
	cudaMalloc((void **)&device_edge, sizeof(edge)*num_edges);
	cudaMemcpy(device_edge, host_edge, sizeof(edge)*num_edges, cudaMemcpyHostToDevice);
	//allocate cost array that will hold cost of each potential edge
	int *cost;
	cudaMalloc((void **)&cost, sizeof(int) * num_edges);
	//while we have not visited all the nodes
	while (num_added != num_nodes - 1) { //O(n)
		int blockSize = getBlockSize();
		int numBlocks = (num_edges + blockSize - 1) / blockSize;
		//kernel call to compute each edge's neighbors cost if added to cluster O(1)
		compute_edge_dist << <numBlocks, blockSize >> > (num_edges, cost, device_graph, device_edge);

		//kernel call to pick minimum cost edge's neighbor O(1)
		int edge_index = min_val(num_edges, cost);//lowest possible edge addition

		//Kernel to update costs of node and its neighbors O(1)
		int node_to_add = edges[edge_index].neighbor;
		//printf("adding node: %d with cost %d\n", node_to_add,new_cost_of_node);
		int number_of_edges_to_update = graph[edges[edge_index].neighbor].neighbors.size();
		edge * edges_to_update = &(graph[edges[edge_index].neighbor].neighbors[0]);
		edge* device_edges_to_update;
		cudaMalloc((void **)&device_edges_to_update, sizeof(edge)*number_of_edges_to_update);
		cudaMemcpy(device_edges_to_update, edges_to_update, sizeof(edge)*number_of_edges_to_update, cudaMemcpyHostToDevice);
		numBlocks = ((number_of_edges_to_update + 1) + blockSize - 1) / blockSize;
		update_costs << <numBlocks, blockSize >> > (edge_index, number_of_edges_to_update, cost, device_graph, device_edge, device_edges_to_update);
		cudaFree(device_edges_to_update);
		//find next lowest cost
		num_added++;
	}
	cudaMemcpy(host_graph, device_graph, sizeof(node)*num_nodes, cudaMemcpyDeviceToHost);
	printResults(graph);
	return graph;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		PFERROR_EXIT(1, "No input file\n");

	ifstream infile(argv[1]);
	if (!infile)
		PFERROR_EXIT(1, "Error opening input file\n");
	string token;
	getline(infile, token, ' ');
	int num_node = atoi(token.c_str());
	getline(infile, token);
	int num_edge = atoi(token.c_str());

	vector<node> graph(num_node);
	vector<edge> edges(2 * num_edge);
	for (int i = 0; i < num_edge; i++) {
		getline(infile, token, ' ');
		int a = atoi(token.c_str());
		getline(infile, token, ' ');
		int b = atoi(token.c_str());
		getline(infile, token);
		int weight = atoi(token.c_str());
		graph[a].neighbors.push_back(edge(weight, b, a));
		graph[b].neighbors.push_back(edge(weight, a, b));
		edges.push_back(edge(weight, b, a));
		edges.push_back(edge(weight, a, b));
	}
	clock_t start = clock();
	vector<node> graphParallel = SPA_parallel(graph, edges, 0);
	clock_t end = clock();
	cout << "Parallel shortest distance algorithm took " << (end - start) << " clock cycles\n\n";

	start = clock();
	vector<node> graphSerial = SPA_serial(graph, 0); // find shortest distance from node 0 to all others
	end = clock();
	cout << "Serial shortest distance algorithm took " << (end - start) << " clock cycles\n\n";

	compareResults(graphParallel, graphSerial);
	return 0;
}
