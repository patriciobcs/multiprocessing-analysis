CPP = /opt/homebrew/opt/llvm/bin/clang++
CPPFLAGS = -fopenmp
NUM_THREADS = 2
NUM_STEPS = (10000)

pi:
	echo "method,num_steps,num_threads,result,runtime" > pi.csv
	$(CPP) $(CPPFLAGS) -o pi pi.cpp
	for type in critical atomic reduction split ; do \
		for threads in 1 2 4 8 ; do \
			for (( c=10000; c<=10000000; c*=10 )) ; do \
				for (( i=0; i<10; i++ )) ; do \
					./pi -num_steps $$c -$$type -threads $$threads; \
				done ; \
			done ; \
		done ; \
	done ; \

vector:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > vector.csv
	$(CPP) $(CPPFLAGS) -o vector vector.cpp
	for i in 2 4 8 10 12 14 15 ; do \
		./vector -N $$i -M $$(expr $$i - 1); \
	done ; \

vector_omp:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > vector_omp.csv
	$(CPP) $(CPPFLAGS) -o vector_omp vector_omp.cpp
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./vector_omp -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

vector_simd:
	echo "num_threads,nrepeat,N,M,Gbytes,time" > vector_simd.csv
	$(CPP) $(CPPFLAGS) -o vector_simd vector_simd.cpp
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./vector_simd -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

fib:
	echo "num_threads,N,time" > fib.csv
	$(CPP) $(CPPFLAGS) -o fib fib.cpp
	for threads in 1 2 4 8 ; do \
		./fib -threads $$threads; \
	done ; \

matrix_mul:
	$(CPP) $(CPPFLAGS) -o matrix_mul matrix_mul.cpp
	./matrix_mul