CC = icc
CFLAGS = -xCORE-AVX512 -qopt-zmm-usage=high -Ofast -Wall -qopenmp -DVECALIGN=64 -g -I/opt/intel/oneapi/advisor/2022.1.0/include/ #-qopt-report=4  -ipo -fno-alias -align -fbuiltin -finline -finline-functions -g -fno-exceptions -simd -simd-function-pointers -qopt-multiple-gather-scatter-by-shuffles -qopt-streaming-stores=auto -qopt-subscript-in-range -vec-threshold=0 -I"${MKLROOT}/include"
#-I/opt/intel/oneapi/advisor/2022.1.0/include/
#-qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout

LDFLAGS = -qopenmp -g -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

objects = kernels.o utils.o
targets = PaLD_test PaLD_par_test PaLD_allz_par_test PaLD_vtune PaLD_triplet_test PaLD_triplet_l2blocking_test PaLD_triplet_par_test PaLD_sgemm_test PaLD_triplet_benchmark PaLD_triplet_omp_benchmark PaLD_allz_benchmark PaLD_allz_omp_benchmark PaLD_allz_omp_explicitnuma_benchmark

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

PaLD_triplet_l2blocking_test : PaLD_triplet_l2blocking_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_allz_par_test : PaLD_allz_par_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_triplet_par_test : PaLD_triplet_par_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_triplet_benchmark : PaLD_triplet_benchmark.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_triplet_omp_benchmark : PaLD_triplet_omp_benchmark.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_allz_benchmark : PaLD_allz_benchmark.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_allz_omp_benchmark : PaLD_allz_omp_benchmark.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_allz_omp_explicitnuma_benchmark : PaLD_allz_omp_explicitnuma_benchmark.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

PaLD_sgemm_test : PaLD_sgemm_test.o $(objects)
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY : clean
clean :
	rm -rf $(objects) $(targets) *~ *.o


