# EE361C final project: Dijkstra's shortest path algorithm
Pathfinder.cu contains both a serial and parallel (GPU) shortest path implementation.

## Running the tests
The master branch may not compile on maverick due to c++ 11 constraints. If this is the case, run the non-C++11 branch code.

Compile pathfinder.cu (you may need to add flags to compile) 
```
nvcc pathfinder.cu
```
Run the executable with a testcase using the format `./a.out {testcase}`:
```
./a.out 10
```

