CXX := g++
CXX_FLAGS := -g -std=c++17

SRC := ./src
INCLUDE := ./include
LIB := ./lib

LIBRARIES := -lglad -lglfw3dll -lglfw3
EXECUTABLE := main

all:./$(EXECUTABLE)

run: all
	./$(EXECUTABLE)

$(EXECUTABLE):$(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ $(LIBRARIES) -o $@