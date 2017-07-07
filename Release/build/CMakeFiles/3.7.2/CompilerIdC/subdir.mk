################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/CMakeFiles/3.7.2/CompilerIdC/CMakeCCompilerId.c 

OBJS += \
./build/CMakeFiles/3.7.2/CompilerIdC/CMakeCCompilerId.o 

C_DEPS += \
./build/CMakeFiles/3.7.2/CompilerIdC/CMakeCCompilerId.d 


# Each subdirectory must supply rules for building sources it contributes
build/CMakeFiles/3.7.2/CompilerIdC/%.o: ../build/CMakeFiles/3.7.2/CompilerIdC/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "build/CMakeFiles/3.7.2/CompilerIdC" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 -std=c++11 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


