# EE361C final project: Dijkstra's shortest path algorithm
Pathfinder.cu contains both a serial and parallel (GPU) shortest path implementation.

## Running the tests
The master branch may not compile on maverick due to c++ 11 constraints. If this is the case, run the non-C++11 branch code.

If your system has C++ 11 support, run:

```make```

<br>
If the compiler complains, use the source code from the `non-C++11` branch and do the following:

Compile pathfinder.cu (you may need to add flags to compile):

```
nvcc pathfinder.cu -o pathfinder
```
Run the executable with a testcase using the format `./pathfinder {testcase}`:
```
./pathfinder 10
```

The test cases are in the format stated below

```
[# nodes] [# edges]
[idx1] [idx2] [weight] <- edge 0
[idx1] [idx2] [weight] <- edge 1
...
[idx1] [idx2] [weight] <- edge n-1
```

To generate test cases, use the generate utility which would put the new test case in a file named the number of nodes

```
./generate <# nodes> <# edges>
./generate 10 10
```
