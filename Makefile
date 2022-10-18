CPP = /opt/homebrew/opt/llvm/bin/clang++
CPPFLAGS = -fopenmp
NUM_THREADS = 2
NUM_STEPS = (10000)

p1:
	echo "method,num_steps,num_threads,result,runtime" > metrics_part_1.csv
	$(CPP) $(CPPFLAGS) -o part_1_pi tp_openmp_part_1_pi.cpp
	for type in critical atomic reduction split ; do \
		for threads in 1 2 4 8 ; do \
			for (( c=10000; c<=10000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./part_1_pi -num_steps $$c -$$type -threads $$threads; \
				done ; \
			done ; \
		done ; \
	done ; \

p2:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > metrics_part_2.csv
	$(CPP) $(CPPFLAGS) -o part_2_vector tp_openmp_part_2_vector.cpp
	for N in 2 4 8 10 12 14 16 ; do \
			for M in 1 3 7 9 11 13 15 ; do \
					./part_2_vector -N $$N -M $$M; \
			done ; \
	done ; \

p2omp:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > metrics_part_2_omp.csv
	$(CPP) $(CPPFLAGS) -o part_2_vector_omp tp_openmp_part_2_vector_omp.cpp
	for threads in 1 2 4 8 ; do \
		for N in 2 4 8 10 12 14 16 ; do \
			for M in 1 3 7 9 11 13 15 ; do \
					./part_2_vector_omp -N $$N -M $$M -threads $$threads; \
			done ; \
		done ; \
	done ; \

p2simd:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > metrics_part_2_simd.csv
	$(CPP) $(CPPFLAGS) -o part_2_vector_simd tp_openmp_part_2_vector_simd.cpp
	for threads in 1 2 4 8 ; do \
		for N in 2 4 8 10 12 14 16 ; do \
			for M in 1 3 7 9 11 13 15 ; do \
					./part_2_vector_simd -N $$N -M $$M -threads $$threads; \
			done ; \
		done ; \
	done ; \

p3:
	$(CPP) $(CPPFLAGS) -o part_3_fib tp_openmp_part_3_fib.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_3_fib

p4:
	$(CPP) $(CPPFLAGS) -o part_4_matrix_mul tp_openmp_part_4_matrix_mul.cpp
	export OMP_NUM_THREADS=$(NUM_THREADS); ./part_4_matrix_mul