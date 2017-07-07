################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../build/CMakeFiles/3.7.2/CompilerIdCXX/CMakeCXXCompilerId.cpp 

OBJS += \
./build/CMakeFiles/3.7.2/CompilerIdCXX/CMakeCXXCompilerId.o 

CPP_DEPS += \
./build/CMakeFiles/3.7.2/CompilerIdCXX/CMakeCXXCompilerId.d 


# Each subdirectory must supply rules for building sources it contributes
build/CMakeFiles/3.7.2/CompilerIdCXX/%.o: ../build/CMakeFiles/3.7.2/CompilerIdCXX/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "build/CMakeFiles/3.7.2/CompilerIdCXX" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


