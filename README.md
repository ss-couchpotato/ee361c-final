# EE361C final project: Dijkstra's shortest path algorithm
Pathfinder.cu contains both a serial and parallel (GPU) shortest path implementation.

## Time Complexity
Serial: O(V^2)

Parallel: O(Log(V * Log(V)))
  
## Running The Tests
The master branch may not compile on maverick due to c++ 11 constraints. If this is the case, run the non-C++11 branch code.

If your system has C++ 11 support, run:

```
make
```
<br>
If the compiler complains, use the source code from the `non-C++11` branch and try compiling.

Next, run the executable with a testcase using the format `./pathfinder {testcase}`:
```
./pathfinder 10
```

### Generating Test Cases
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
