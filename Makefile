# Compiler
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -O3 -Wall -Iinclude  -fopenmp -march=native -funroll-loops -MMD -MP
NVCCFLAGS := -O3 -Xcompiler -fopenmp -Iinclude

# Folder Paths
SRC_DIR := src
OBJ_DIR := obj
INC_DIR := include
BIN_DIR := bin

# Source files
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Object files
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# All object files
OBJ_FILES := $(CPP_OBJS) $(CU_OBJS)

# Dependency files
CPP_DEPS := $(CPP_OBJS:.o=.d)
CU_DEPS := $(CU_OBJS:.o=.d)
DEP_FILES := $(CPP_DEPS) $(CU_DEPS)

# Target executable
TARGET := $(BIN_DIR)/run

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) -o $@ $^ -Xcompiler -fopenmp

# Compile C++ files with dependency generation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files and generate dependencies manually
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -M $< -Iinclude > $(OBJ_DIR)/$*.d
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Include dependency files if they exist
-include $(DEP_FILES)

# Phony targets
.PHONY: all clean
