################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../build/CMakeFiles/feature_tests.cxx 

C_SRCS += \
../build/CMakeFiles/feature_tests.c 

CXX_DEPS += \
./build/CMakeFiles/feature_tests.d 

OBJS += \
./build/CMakeFiles/feature_tests.o 

C_DEPS += \
./build/CMakeFiles/feature_tests.d 


# Each subdirectory must supply rules for building sources it contributes
build/CMakeFiles/%.o: ../build/CMakeFiles/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "build/CMakeFiles" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

build/CMakeFiles/%.o: ../build/CMakeFiles/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "build/CMakeFiles" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


