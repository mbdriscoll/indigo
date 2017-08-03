default: _customgpu.o

%.o: %.cu
	nvcc -arch=sm_61 -Xcompiler '-fPIC' -rdc=true -c -o temp.o $<
	nvcc -arch=sm_61 -Xcompiler '-fPIC' -dlink -o $@ temp.o -lcudart
	ar cru libgpu.a $@ temp.o
	ranlib libgpu.a

foo:
	#nvcc -arch=sm_35 -Xcompiler '-fPIC' -dc $<
	#nvcc -arch=sm_35 -Xcompiler '-fPIC' -dlink $@ -o $@
	#nvcc -Xcompiler -fPIC -x cu -arch=sm_30 -I. -dc $< -o $@
