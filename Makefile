CC = nvcc
GCC = gcc
CCSTD = c++03
NVCCFLAG = -arch=compute_35 -code=sm_35 -lcublas
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
