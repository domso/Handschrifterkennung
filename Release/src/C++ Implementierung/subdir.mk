################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/C++\ Implementierung/main.cu 

CPP_SRCS += \
../src/C++\ Implementierung/Layer.cpp \
../src/C++\ Implementierung/NeuronalNetwork.cpp \
../src/C++\ Implementierung/Node.cpp 

OBJS += \
./src/C++\ Implementierung/Layer.o \
./src/C++\ Implementierung/NeuronalNetwork.o \
./src/C++\ Implementierung/Node.o \
./src/C++\ Implementierung/main.o 

CU_DEPS += \
./src/C++\ Implementierung/main.d 

CPP_DEPS += \
./src/C++\ Implementierung/Layer.d \
./src/C++\ Implementierung/NeuronalNetwork.d \
./src/C++\ Implementierung/Node.d 


# Each subdirectory must supply rules for building sources it contributes
src/C++\ Implementierung/Layer.o: ../src/C++\ Implementierung/Layer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -odir "src/C++ Implementierung" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/C++\ Implementierung/NeuronalNetwork.o: ../src/C++\ Implementierung/NeuronalNetwork.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -odir "src/C++ Implementierung" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/C++\ Implementierung/Node.o: ../src/C++\ Implementierung/Node.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -odir "src/C++ Implementierung" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/C++\ Implementierung/main.o: ../src/C++\ Implementierung/main.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -odir "src/C++ Implementierung" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


