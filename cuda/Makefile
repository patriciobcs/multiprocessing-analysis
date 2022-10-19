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
	for threads in 1 2 4 8 ; do \
		for i in 2 4 8 10 12 14 15 ; do \
			./vector -N $$i -M $$(expr $$i - 1) -threads $$threads; \
		done ; \
	done ; \

matrix_mul:
	echo "num_threads,N,time" > matrix_mul.csv
	$(CPP) $(CPPFLAGS) -o part_3_fib matrix_mul.cpp
	for threads in 1 2 4 8 ; do \
		./matrix_mul -threads $$threads; \
	done ; \