CC = nvcc
GCC = gcc
CCSTD = c++11
NVCCFLAG = -lcublas
BIN = pathfinder
GENERATE = generate

all: $(BIN) $(GENERATE)

$(GENERATE): %: %.c
	$(GCC) -o $@ $<

$(BIN): %: %.cu
	$(CC) $(NVCCFLAG) -o $@ -std $(CCSTD) $<

.PHONY: clean

clean:
	rm -f $(BIN)
