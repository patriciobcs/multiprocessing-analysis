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
	for i in 2 4 8 10 12 14 15 ; do \
		./part_2_vector -N $$i -M $$(expr $$i - 1); \
	done ; \

p2omp:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > metrics_part_2_omp.csv
	$(CPP) $(CPPFLAGS) -o part_2_vector_omp tp_openmp_part_2_vector_omp.cpp
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./part_2_vector_omp -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

p2simd:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > metrics_part_2_simd.csv
	$(CPP) $(CPPFLAGS) -o part_2_vector_simd tp_openmp_part_2_vector_simd.cpp
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./part_2_vector_simd -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

p3:
	echo "num_threads,N,time" > metrics_part_3.csv
	$(CPP) $(CPPFLAGS) -o part_3_fib tp_openmp_part_3_fib.cpp
	for threads in 1 2 4 8 ; do \
		./part_3_fib -threads $$threads; \
	done ; \

p4:
	$(CPP) $(CPPFLAGS) -o part_4_matrix_mul tp_openmp_part_4_matrix_mul.cpp
	./part_4_matrix_mul