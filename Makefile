# Used both locally and by colab.ipynb. Everything generated (compiled programs, profiler reports) goes into build/.
# Git keeps the empty folder (build/.gitkeep) but ignores what's inside it.
# Programs are only recompiled when their .cu file changes (or when TILE changes,
# since each tile size gets its own binary).
#
#   make run          compile and run once
#   make sweep        run for N = 1024 ... 16384 (the sizes Q1 asks for)
#   make tilesweep    rebuild and run with several tile sizes, to pick the best one
#   make profile      time breakdown (Nsight Systems)
#   make ncu          kernel detail: throughput, memory, occupancy (Nsight Compute)
#   make occupancy    Achieved Occupancy only (Nsight Compute)
#   make clean        empty the build/ folder
#
# Sources live in assignment/. Other program, arguments or tile size:
#   make run PROG=matmul ARGS="1024"     (compiles assignment/matmul.cu)
#   make run TILE=128 ARGS=8192

PROG  ?= hw2-convolution
ARGS  ?= 1024
TILE  ?= 256
SRC   := assignment
BUILD := build
BIN   := $(BUILD)/$(PROG)_tile$(TILE)

SIZES := 1024 2048 4096 8192 16384
TILES := 32 64 128 256 512 1024

.PHONY: run sweep tilesweep profile ncu occupancy clean

# Compile assignment/X.cu into build/X_tileT, with TILE_WIDTH set from the command line
$(BIN): $(SRC)/$(PROG).cu | $(BUILD)
	nvcc -arch=native -DTILE_WIDTH=$(TILE) -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

run: $(BIN)
	./$(BIN) $(ARGS)

sweep: $(BIN)
	@for n in $(SIZES); do echo "=== N = $$n, TILE = $(TILE) ==="; ./$(BIN) $$n; done

tilesweep:
	@for t in $(TILES); do echo "=== TILE = $$t ==="; $(MAKE) --no-print-directory run TILE=$$t; done

profile: $(BIN)
	nsys profile --stats=true --force-overwrite true -o $(BUILD)/$(PROG)-report ./$(BIN) $(ARGS)

ncu: $(BIN)
	ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats ./$(BIN) $(ARGS)

occupancy: $(BIN)
	ncu --section Occupancy ./$(BIN) $(ARGS)

# Deletes everything in build/ except the hidden .gitkeep
clean:
	rm -rf $(BUILD)/*
