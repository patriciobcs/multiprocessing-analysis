NVCC = nvcc
NUM_THREADS = 2
$(eval SHELL:=/bin/bash)

pi: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi.csv
	$(NVCC) -o pi pi.cu
	for threads in 1 32 64 128 256 ; do \
		for thread_steps in 1 64 256 1024 ; do \
			for (( c=1000000; c<=100000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi -N $$c -t $$threads -ts $$thread_steps; \
				done ; \
			done ; \
		done ; \
	done ; \

pi_atomic: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi_atomic.csv
	$(NVCC) -o pi_atomic pi_atomic.cu
	for threads in 1 32 64 128 256 ; do \
		for thread_steps in 1 64 256 1024 ; do \
			for (( c=1000000; c<=100000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi_atomic -N $$c -t $$threads -ts $$thread_steps; \
				done ; \
			done ; \
		done ; \
	done ; \

pi_msr: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi_msr.csv
	$(NVCC) -o pi_msr pi_msr.cu
	for threads in 1 32 64 128 256 ; do \
		for thread_steps in 1 64 256 1024 ; do \
			for (( c=1000000; c<=100000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi_msr -N $$c -t $$threads -ts $$thread_steps; \
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