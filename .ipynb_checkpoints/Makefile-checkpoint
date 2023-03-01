CXX=icpx
CXXFLAGS=-fsycl -fopenmp -Wall -O2 -g
LDFLAGS=
SRC=./src/main.cpp ./src/io_routines.cpp ./src/stegano_routines.cpp
OBJ=$(SRC:.cpp=.o)
TARGET=./builds/stegano.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET)
	rm -f $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean