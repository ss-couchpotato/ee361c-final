#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

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
		int weight;
		int neighbor;

	edge(int weight, int neighbor) {
		this->weight = weight;
		this->neighbor = neighbor;
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
void SPA_serial(vector<node> &graph, int start_node) 
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
	} 
	printResults(graph);
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


int main(int argc, char* argv[])
{
	if (argc < 2)
		PFERROR_EXIT(1, "No input file\n");
	
	ifstream infile(argv[1]);
	if (!infile)
		PFERROR_EXIT(1, "Error opening input file\n");
	string token;
	getline(infile, token, ' ');
	int num_node = stoi(token);
	getline(infile, token);
	int num_edge = stoi(token);

	vector<node> graph(num_node);
	for (int i = 0; i < num_edge; i++) {
		getline(infile, token, ' ');
		int a = stoi(token);
		getline(infile, token, ' ');
		int b = stoi(token);
		getline(infile, token);
		int weight = stoi(token);
		graph[a].neighbors.push_back(edge(weight, b));
		graph[b].neighbors.push_back(edge(weight, a));
	}

	clock_t start = clock();
	SPA_serial(graph, 0); // find shortest distance from node 0 to all others
	clock_t end = clock();
	cout << "Serial shortest distance algorithm took " << (end - start) << " clock cycles\n\n";

	return 0;
}
