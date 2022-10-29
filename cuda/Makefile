NVCC = nvcc
NUM_THREADS = 2
$(eval SHELL:=/bin/bash)

pi: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi.csv
	$(NVCC) -o pi pi.cu
	for thread_steps in 1 64 256 1024 ; do \
		for (( c=1000000; c<=1000000000; c*=10 )) ; do \
			for (( i=0; i<10; i++ )) ; do \
				./pi -N $$c -ts $$thread_steps; \
			done ; \
		done ; \
	done ; \

pi_mt: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi_mt.csv
	$(NVCC) -o pi_mt pi_mt.cu
	for threads in 1 32 64 128 256 ; do \
		for thread_steps in 1 64 256 1024 ; do \
			for (( c=1000000; c<=1000000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi_mt -N $$c -t $$threads -ts $$thread_steps; \
				done ; \
			done ; \
		done ; \
	done ; \

pi_atomic: FORCE
	echo "num_steps,num_blocks,num_threads,thread_steps,result,runtime" > pi_atomic.csv
	$(NVCC) -o pi_atomic pi_atomic.cu
	for threads in 1 32 64 128 256 ; do \
		for thread_steps in 1 64 256 1024 ; do \
			for (( c=1000000; c<=1000000000; c*=10 )) ; do \
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
			for (( c=1000000; c<=1000000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi_msr -N $$c -t $$threads -ts $$thread_steps; \
				done ; \
			done ; \
		done ; \
	done ; \

vector: FORCE
	echo "num_threads,N,M,Gbytes,time" > vector.csv
	$(NVCC) -o vector vector.cu
	for i in 2 4 8 10 12 14 15 ; do \
		for (( t=0; t<10; t++ )) ; do \
			./vector -N $$i -M $$(expr $$i - 1); \
		done ; \
	done ; \

vector_atomic: FORCE
	echo "num_threads,N,M,Gbytes,time" > vector_atomic.csv
	$(NVCC) -o vector_atomic vector_atomic.cu
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			for (( t=0; t<10; t++ )) ; do \
				./vector_atomic -N $$i -M $$(expr $$i - 1) -t $$threads; \
			done ; \
		done ; \
	done ; \

vector_shared: FORCE
	echo "num_threads,N,M,Gbytes,time" > vector_shared.csv
	$(NVCC) -o vector_shared vector_shared.cu
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			for (( t=0; t<10; t++ )) ; do \
				./vector_shared -N $$i -M $$(expr $$i - 1) -t $$threads; \
			done ; \
		done ; \
	done ; \

FORCE: ;