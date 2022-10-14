CPP = /opt/homebrew/opt/llvm/bin/clang
CPPFLAGS = -fopenmp
NUM_THREADS = 2

p1:
	$(CPP) $(CPPFLAGS) -o part_1_pi tp_openmp_part_1_pi.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_1_pi

p2:
	$(CPP) $(CPPFLAGS) -o part_2_vector tp_openmp_part_2_vector.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_2_vector

p3:
	$(CPP) $(CPPFLAGS) -o part_3_fib tp_openmp_part_3_fib.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_3_fib

p4:
	$(CPP) $(CPPFLAGS) -o part_4_matrix_mul tp_openmp_part_4_matrix_mul.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_4_matrix_mul