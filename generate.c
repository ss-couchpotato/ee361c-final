#include "stdio.h"
#include "stdlib.h"

#define PFERROR_EXIT(exit_code, ...)                                           \
  do {                                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
    exit(exit_code);                                                           \
  } while (0)

#define MAX_WEIGHT 100

int main(int argc, char* argv[]) {
  if (argc != 3)
    PFERROR_EXIT(1, "Usage: ./generate <# nodes> <# edges>\n");

  int n = atoi(argv[1]);
  int m = atoi(argv[2]);
  if (n <= 1 || m <= 0) {
    fprintf(stderr, "# of nodes must be greater than 1\n");
    PFERROR_EXIT(1, "Usage: ./generate <# nodes> <# edges>\n");
  }

  FILE *file = fopen(argv[1], "w");
  printf("%d %d\n", n, m);
  fprintf(file, "%d %d\n", n, m);
  for (int i = 0; i < m; i++) {
    int a = rand() % n;
    int b;
    do {
      b = rand() % n;
    } while (a == b);
    fprintf(file, "%d %d %d\n", a, b, rand() % MAX_WEIGHT);
  }
  fclose(file);
}
