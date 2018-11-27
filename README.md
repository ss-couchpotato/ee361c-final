# EE361C final project: Dijkstra's shortest path algorithm
Pathfinder.cu contains both a serial and parallel (GPU) shortest path implementation.

## Running the tests
If your system has C++ 11 support, run:

```make```

<br>
If the compiler complains, use the source code from the `non-C++11` branch and do the following:

Compile pathfinder.cu (you may need to add flags to compile):

```
nvcc pathfinder.cu
```
Run the executable with a testcase using the format `./a.out {testcase}`:
```
./a.out 10
```

