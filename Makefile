# Build mode
# 0: development (max safety, no optimisation)
# 1: release (min safety, optimisation)

BUILD_MODE?=0

# Path to PBMake

PATH_PBMAKE=../PBMake

# Compiler arguments depending on BUILD_MODE

ifeq ($(BUILD_MODE), 0)
	BUILD_ARG=-I$(PATH_PBMAKE)/Include -Wall -Wextra -Og -ggdb -g3 -DPBERRALL='1' \
	  -DBUILDMODE=$(BUILD_MODE)
	LINK_ARG=-L$(PATH_PBMAKE)/Lib -lpbdev -lm -rdynamic
else 
  ifeq ($(BUILD_MODE), 1)
	  BUILD_ARG=-I$(PATH_PBMAKE)/Include -Wall -Wextra -Werror -Wfatal-errors -O3 \
		  -DPBERRSAFEMALLOC='1' -DPBERRSAFEIO='1' -DBUILDMODE=$(BUILD_MODE)
	  LINK_ARG=-L$(PATH_PBMAKE)/Lib -lpbrelease -lm -rdynamic
	endif
endif

# Compiler

COMPILER=gcc-7


GTKPARAM=`pkg-config --cflags gtk+-3.0`
GTKLINK=`pkg-config --libs gtk+-3.0`
CAIROPARAM=`pkg-config --cflags cairo`
CAIROLINK=`pkg-config --libs cairo`

# Rules for the executable

all: clean main

main: main.o Makefile
	$(COMPILER) main.o $(LINK_ARG) $(GTKLINK) $(CAIROLINK) -o main 

main.o: main.c Makefile
	$(COMPILER) $(BUILD_ARG) $(GTKPARAM) $(CAIROPARAM) -c main.c 

clean:
	rm -f *.o main

runTest01:
	main -encode test01.tga -verbose
