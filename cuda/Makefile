NVCC = nvcc
NUM_THREADS = 2
$(eval SHELL:=/bin/bash)

pi: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi.csv
	$(NVCC) -o pi pi.cu
	for threads in 1 32 64 128 256 ; do \
		for threads in 1 64 256 1024 ; do \
			for (( c=1000000; c<=100000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi -num_steps $$c -threads $$threads; \
				done ; \
			done ; \
		done ; \
	done ; \

vector: FORCE
	echo "num_threads,nrepeat,N,M,Gbytes,time" > vector.csv
	$(CPP) $(CPPFLAGS) -o vector vector.cpp
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./vector -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

matrix_mul: FORCE
	echo "num_threads,N,time" > matrix_mul.csv
	$(CPP) $(CPPFLAGS) -o part_3_fib matrix_mul.cpp
	for threads in 1 2 4 8 ; do \
		./matrix_mul -threads $$threads; \
	done ; \

FORCE: ;