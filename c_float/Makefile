CC = icc
CFLAGS = -Ofast -Wall -ipo -qopenmp -xCORE-AVX512 -qopt-zmm-usage=high -DEBUG -DVECALIGN=64 -g
LDFLAGS = -qopenmp -ipo -DEBUG -g

objects = kernels.o utils.o
targets = PaLD_test PaLD_par_test PaLD_vtune

.PHONY : default
default: $(targets)

.PHONY : all
all: clean $(targets)

PaLD_vtune : PaLD_vtune.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_test : PaLD_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS) 

PaLD_par_test : PaLD_par_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS) 

%.o : %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean
clean :
	rm -rf $(objects) $(targets) *~ *.o


