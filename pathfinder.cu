#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <sstream>
#include <vector>

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

	return 0;
}
