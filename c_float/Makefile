CC = icc
CFLAGS = -Ofast -Wall -ipo -qopenmp -mtune=cascadelake -xCORE-AVX512 -qopt-zmm-usage=high -DEBUG -DVECALIGN=64 -fno-alias -align -fbuiltin -finline -finline-functions -qopt-report-phase=vec -qopt-report=5  -qopt-report-file=vecreport.optrpt -g -unroll8 -fno-exceptions -simd -simd-function-pointers -qopt-multiple-gather-scatter-by-shuffles -qopt-streaming-stores=auto -qopt-subscript-in-range -vec-threshold=0

LDFLAGS = -qopenmp -DEBUG -g 

objects = kernels.o utils.o
targets = PaLD_test PaLD_par_test PaLD_vtune PaLD_triplet_test PaLD_triplet_par_test

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

PaLD_triplet_test : PaLD_triplet_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS) 

PaLD_triplet_par_test : PaLD_triplet_par_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS) 

%.o : %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean
clean :
	rm -rf $(objects) $(targets) *~ *.o


