default: _customgpu.o

%.o: %.cu %.h
	nvcc -arch=sm_61 -Xcompiler '-fPIC' -rdc=true -c -o temp.o $<
	nvcc -arch=sm_61 -Xcompiler '-fPIC' -dlink -o $@ temp.o -lcudart
	ar cru libgpu.a $@ temp.o
	ranlib libgpu.a
