#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
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
		vector<edge> neighbors;

	node() {
		cost = INT_MAX;
		parent = -1;
	}
};

void printResults(int distance[], int num_nodes) 
{ 
   cout << "Vertex\t Distance from start_node\n"; 
   for (int i = 0; i < num_nodes; i++) {
      cout << i << "\t" << distance[i] << "\n"; 
   }
} 

// helper function that finds nearest neighbor for serial SPA
int minDistance(int distance[], bool used_nodes[], int num_nodes) 
{ 
	int min_distance = INT_MAX;
	int min_node = -1; 

	for (int node = 0; node < num_nodes; node++) {
		if (used_nodes[node] == false && distance[node] <= min_distance) {
			min_distance = distance[node], min_node = node; 
		}
	}     
	return min_node; 
} 

// returns weight of connection between node1 and node2 if they are neighbors
int areNeighbors(vector<node> graph, int node1, int node2) 
{
	vector<edge> neighbors = graph[node1].neighbors;
	for (int i = 0; i < neighbors.size(); i++) {
		if (neighbors[i].neighbor == node2) {
			return neighbors[i].weight;
		}
	}
	return -1;
}

// serial implementation of Dijkstra's shortest path algorithm
void SPA_serial(vector<node> graph, int start_node) 
{ 
	int num_nodes = graph.size();
	int distance[num_nodes]; // holds distance from start_node to node i
	bool used_nodes[num_nodes]; // i is true is node is used in shortest path tree

	for (int i = 0; i < num_nodes; i++) {
		used_nodes[i] = false; 
		distance[i] = INT_MAX;
	}
	distance[start_node] = 0; 

	// find shortest path from start_node to each node
	for (int count = 0; count < num_nodes-1; count++) { 
		int nearest_node = minDistance(distance, used_nodes, num_nodes); 
		used_nodes[nearest_node] = true; 

		for (int curr_node = 0; curr_node < num_nodes; curr_node++) {
			int weight = areNeighbors(graph, nearest_node, curr_node);
			if (!used_nodes[curr_node] && weight != -1 && distance[nearest_node] != INT_MAX 
				&& (distance[nearest_node] + weight) < distance[curr_node]) {
				distance[curr_node] = distance[nearest_node] + weight; 
			}
		}
	} 
	printResults(distance, num_nodes);
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
