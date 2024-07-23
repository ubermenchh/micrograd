CC = gcc 
CFLAGS = -Wall -Wextra -std=c11 -g 
LDFLAGS = -lm 
TARGET = micrograd 

$(TARGET): micrograd.o
	$(CC) $(CFLAGS) -o $(TARGET) micrograd.o $(LDFLAGS)

micrograd.o: micrograd.c micrograd.h 
	$(CC) $(CFLAGS) -c micrograd.c 

debug: $(TARGET)
	gdb ./$(TARGET)

clean: 
	rm -rf $(TARGET) *.o 

.PHONY: clean debug
