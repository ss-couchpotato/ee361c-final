CC = nvcc
CCSTD = c++11
BIN = pathfinder 

all: $(BIN)

$(BIN): %: %.cu
	$(CC) -o $@ -std $(CCSTD) $<

.PHONY: clean

clean:
	rm -f $(BIN)
