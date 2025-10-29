######################################################################
# Compiler Setup
CC      = gcc
NVCC    = nvcc

######################################################################
# Directories
CPU_SRC_CORE      = cpu/src/core
CPU_SRC_FEATURES  = cpu/src/features
CPU_SRC_IO        = cpu/src/io
CPU_INCLUDE_DIR   = cpu/include

GPU_SRC_CORE      = gpu/src/core
GPU_SRC_FEATURES  = gpu/src/features
GPU_SRC_IO        = gpu/src/io
GPU_INCLUDE_DIR   = gpu/include

EXAMPLES_DIR  = examples
BUILD_DIR     = build
TOOLS_DIR     = tools
DATA_DIR      = data/hotel/frames
OUTPUT_DIR    = output
PROFILE_DIR   = profiles

OUTPUT_CPU    = $(OUTPUT_DIR)/cpu
OUTPUT_GPU    = $(OUTPUT_DIR)/gpu
FRAMES_CPU    = $(OUTPUT_CPU)/frames
FRAMES_GPU    = $(OUTPUT_GPU)/frames

# Default values for features and frames
N_FEATURES    = 150
MAX_FRAMES     = 99999999


######################################################################
# Flags
CUDA_ARCH ?= 75             
ARCH       = sm_$(CUDA_ARCH)
FLAG1        = -DNDEBUG
CPU_CFLAGS   = $(FLAG1) -I$(CPU_INCLUDE_DIR)
GPU_CFLAGS   = $(FLAG1) -I$(GPU_INCLUDE_DIR)
GPUFLAGS = -Xcompiler "-fPIC" -I$(GPU_INCLUDE_DIR) \
           "-gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)" \
           "-gencode=arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)" \
           -O3


LIB          = -L/usr/local/lib -L/usr/lib

# CPU object files
OBJS_CPU      = $(BUILD_DIR)/cpu_convolve.o $(BUILD_DIR)/cpu_pyramid.o \
                 $(BUILD_DIR)/cpu_klt.o $(BUILD_DIR)/cpu_klt_util.o \
                 $(BUILD_DIR)/cpu_selectGoodFeatures.o $(BUILD_DIR)/cpu_storeFeatures.o \
                 $(BUILD_DIR)/cpu_trackFeatures.o $(BUILD_DIR)/cpu_writeFeatures.o \
                 $(BUILD_DIR)/cpu_error.o $(BUILD_DIR)/cpu_pnmio.o

# GPU object files  
OBJS_GPU      = $(BUILD_DIR)/gpu_convolve.o $(BUILD_DIR)/gpu_pyramid.o \
                 $(BUILD_DIR)/gpu_klt.o $(BUILD_DIR)/gpu_klt_util.o \
                 $(BUILD_DIR)/gpu_selectGoodFeatures.o $(BUILD_DIR)/gpu_storeFeatures.o \
                 $(BUILD_DIR)/gpu_trackFeatures.o $(BUILD_DIR)/gpu_writeFeatures.o \
                 $(BUILD_DIR)/gpu_error.o $(BUILD_DIR)/gpu_pnmio.o

######################################################################
# Default build
all: $(BUILD_DIR) $(OUTPUT_CPU) $(OUTPUT_GPU) $(PROFILE_DIR) lib cpu gpu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUTPUT_CPU):
	mkdir -p $(FRAMES_CPU)

$(OUTPUT_GPU):
	mkdir -p $(FRAMES_GPU)

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)/cpu $(PROFILE_DIR)/gpu

######################################################################
# Compile object files - CPU
$(BUILD_DIR)/cpu_convolve.o: $(CPU_SRC_CORE)/convolve.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_pyramid.o: $(CPU_SRC_CORE)/pyramid.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_klt.o: $(CPU_SRC_CORE)/klt.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_klt_util.o: $(CPU_SRC_CORE)/klt_util.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_selectGoodFeatures.o: $(CPU_SRC_FEATURES)/selectGoodFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_storeFeatures.o: $(CPU_SRC_FEATURES)/storeFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_trackFeatures.o: $(CPU_SRC_FEATURES)/trackFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_writeFeatures.o: $(CPU_SRC_FEATURES)/writeFeatures.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_error.o: $(CPU_SRC_IO)/error.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@
$(BUILD_DIR)/cpu_pnmio.o: $(CPU_SRC_IO)/pnmio.c
	$(CC) -c $(CPU_CFLAGS) $< -o $@

# Compile object files - GPU
$(BUILD_DIR)/gpu_convolve.o: $(GPU_SRC_CORE)/convolve.cu
	$(NVCC) -c $(GPUFLAGS) $< -o $@
$(BUILD_DIR)/gpu_pyramid.o: $(GPU_SRC_CORE)/pyramid.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_klt.o: $(GPU_SRC_CORE)/klt.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_klt_util.o: $(GPU_SRC_CORE)/klt_util.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_selectGoodFeatures.o: $(GPU_SRC_FEATURES)/selectGoodFeatures.c
	$(NVCC) -c $(GPUFLAGS) $< -o $@
$(BUILD_DIR)/gpu_storeFeatures.o: $(GPU_SRC_FEATURES)/storeFeatures.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_trackFeatures.o: $(GPU_SRC_FEATURES)/trackFeatures.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_writeFeatures.o: $(GPU_SRC_FEATURES)/writeFeatures.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_error.o: $(GPU_SRC_IO)/error.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@
$(BUILD_DIR)/gpu_pnmio.o: $(GPU_SRC_IO)/pnmio.c
	$(CC) -c $(GPU_CFLAGS) $< -o $@

######################################################################
# Build library
lib: lib-cpu

lib-cpu:
	@mkdir -p $(BUILD_DIR)
	$(MAKE) $(OBJS_CPU)
	rm -f libklt.a
	ar ruv libklt.a $(OBJS_CPU)
	@echo "CPU Library built: libklt.a"

lib-gpu:
	@mkdir -p $(BUILD_DIR)
	$(MAKE) $(OBJS_GPU)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(OBJS_GPU)
	@echo "GPU Library built: libklt_gpu.a"

######################################################################
# CPU build & run
cpu: lib-cpu $(OUTPUT_CPU)
	@echo "Building CPU version..."
	$(CC) -O3 $(CPU_CFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_CPU)/"' \
		-DMAX_FRAMES=$(MAX_FRAMES) -DN_FEATURES=$(N_FEATURES) \
		-o main_cpu $(EXAMPLES_DIR)/main_cpu.c -L. -lklt $(LIB) -lm
	@echo "Running CPU version..."
	./main_cpu
	@echo "Creating CPU video..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		ffmpeg -framerate 30 -i "$(FRAMES_CPU)/feat%d.ppm" -c:v libx264 -pix_fmt yuv420p $(OUTPUT_CPU)/video.mp4; \
		echo "CPU video created at $(OUTPUT_CPU)/video.mp4"; \
	else \
		echo "ffmpeg not found - skipping video creation"; \
		echo "CPU frames saved in $(FRAMES_CPU)/"; \
	fi

######################################################################
# GPU build & run
gpu: lib-gpu $(OUTPUT_GPU)
	@echo "Building GPU version..."
	$(NVCC) -O3 $(GPUFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_GPU)/"' \
		-DMAX_FRAMES=$(MAX_FRAMES) -DN_FEATURES=$(N_FEATURES) \
		-o main_gpu $(EXAMPLES_DIR)/main_gpu.c -L. -lklt_gpu $(LIB) -lm
	@echo "Running GPU version..."
	./main_gpu
	@echo "Creating GPU video..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		ffmpeg -framerate 30 -i "$(FRAMES_GPU)/feat%d.ppm" -c:v libx264 -pix_fmt yuv420p $(OUTPUT_GPU)/video.mp4; \
		echo "GPU video created at $(OUTPUT_GPU)/video.mp4"; \
	else \
		echo "ffmpeg not found - skipping video creation"; \
		echo "GPU frames saved in $(FRAMES_GPU)/"; \
	fi

######################################################################
# Compare both
compare: cpu gpu
	@echo "Comparison done — CPU and GPU outputs ready!"
	@echo "CPU: $(OUTPUT_CPU)/video.mp4"
	@echo "GPU: $(OUTPUT_GPU)/video.mp4"

######################################################################
# Profiling CPU
cpu-profile: clean lib-cpu $(OUTPUT_CPU)
	@echo "Profiling CPU version..."
	$(CC) -pg -O3 $(CPU_CFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_CPU)/"' \
		-DMAX_FRAMES=$(MAX_FRAMES) -DN_FEATURES=$(N_FEATURES) \
		-o main_cpu $(EXAMPLES_DIR)/main_cpu.c -L. -lklt $(LIB) -lm
	./main_cpu
	$(eval PROFILE_TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	$(eval CPU_PROF_DIR := $(PROFILE_DIR)/cpu/test_$(PROFILE_TIMESTAMP))
	mkdir -p $(CPU_PROF_DIR)
	mv gmon.out $(CPU_PROF_DIR)/
	gprof ./main_cpu $(CPU_PROF_DIR)/gmon.out > $(CPU_PROF_DIR)/profile.txt
	gprof ./main_cpu $(CPU_PROF_DIR)/gmon.out | python3 $(TOOLS_DIR)/gprof2dot.py -s -o $(CPU_PROF_DIR)/profile.dot
	dot -Tpdf $(CPU_PROF_DIR)/profile.dot -o $(CPU_PROF_DIR)/profile.pdf
	@echo "CPU profiling complete: $(CPU_PROF_DIR)/profile.pdf"

######################################################################
# Profiling GPU
gpu-profile: clean lib-gpu $(OUTPUT_GPU)
	@echo "Profiling GPU version..."
	$(NVCC) -pg -O3 $(GPUFLAGS) -DDATA_DIR='"$(DATA_DIR)/"' -DOUTPUT_DIR='"$(FRAMES_GPU)/"' \
		-DMAX_FRAMES=$(MAX_FRAMES) -DN_FEATURES=$(N_FEATURES) \
		-o main_gpu $(EXAMPLES_DIR)/main_gpu.c -L. -lklt_gpu $(LIB) -lm
	./main_gpu
	$(eval PROFILE_TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	$(eval GPU_PROF_DIR := $(PROFILE_DIR)/gpu/test_$(PROFILE_TIMESTAMP))
	mkdir -p $(GPU_PROF_DIR)
	mv gmon.out $(GPU_PROF_DIR)/
	gprof ./main_gpu $(GPU_PROF_DIR)/gmon.out > $(GPU_PROF_DIR)/profile.txt
	gprof ./main_gpu $(GPU_PROF_DIR)/gmon.out | python3 $(TOOLS_DIR)/gprof2dot.py -s -o $(GPU_PROF_DIR)/profile.dot
	dot -Tpdf $(GPU_PROF_DIR)/profile.dot -o $(GPU_PROF_DIR)/profile.pdf
	@echo "GPU profiling complete: $(GPU_PROF_DIR)/profile.pdf"

######################################################################
# Cleaning
clean:
	@echo "Cleaning build files and outputs..."
	rm -f $(BUILD_DIR)/*.o *.a main_cpu main_gpu *.tar *.tar.gz libklt.a libklt_gpu.a \
	      feat*.ppm features.ft features.txt
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Cleaned build and output directories."

clean-all:
	@echo "🔥 Deep cleaning everything..."
	rm -f $(BUILD_DIR)/*.o *.a main_cpu main_gpu *.tar *.tar.gz libklt.a libklt_gpu.a \
	      feat*.ppm features.ft features.txt
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR) $(PROFILE_DIR)
	@echo "✓ All cleaned up — fresh start!"

######################################################################
# Change data directory path
path:
	@echo "=========================================="
	@echo "Available Datasets:"
	@echo "=========================================="
	@if [ -d "data" ]; then \
		index=1; \
		for dataset in data/*/frames; do \
			if [ -d "$$dataset" ]; then \
				dataset_name=$$(basename $$(dirname "$$dataset")); \
				echo ""; \
				echo "$$index) Dataset: $$dataset_name"; \
				echo "   Path: $$dataset"; \
				img_count=$$(ls "$$dataset"/img*.pgm 2>/dev/null | wc -l); \
				if [ $$img_count -gt 0 ]; then \
					first_img=$$(ls "$$dataset"/img*.pgm 2>/dev/null | head -1); \
					if [ -f "$$first_img" ]; then \
						resolution=$$(file "$$first_img" | grep -o '[0-9]* x [0-9]*' | head -1); \
						if [ -n "$$resolution" ]; then \
							echo "   Resolution: $$resolution"; \
						else \
							echo "   Resolution: Unknown"; \
						fi; \
					else \
						echo "   Resolution: Unknown"; \
					fi; \
					echo "   Frames: $$img_count"; \
				else \
					echo "   Frames: 0 (no images found)"; \
				fi; \
				index=$$((index + 1)); \
			fi; \
		done; \
		echo ""; \
		echo "=========================================="; \
		echo "Current DATA_DIR: $(DATA_DIR)"; \
		echo "=========================================="; \
		echo ""; \
		echo "$$index) Custom Path (enter full path)"; \
		echo ""; \
		echo "Select dataset by index (1-$$index) or enter full path:"; \
		read -p "Choice: " choice; \
		if echo "$$choice" | grep -q '^[0-9]\+$$'; then \
			if [ $$choice -ge 1 ] && [ $$choice -lt $$index ]; then \
				selected_path=$$(ls -d data/*/frames | sed -n "$$choice p"); \
				selected_name=$$(basename $$(dirname "$$selected_path")); \
				echo "Selected: $$selected_name ($$selected_path)"; \
				sed -i 's|^DATA_DIR.*|DATA_DIR      = '"$$selected_path"'|' Makefile; \
				echo "DATA_DIR updated to: $$selected_path"; \
				echo "Updated Makefile:"; \
				grep "^DATA_DIR" Makefile; \
			elif [ $$choice -eq $$index ]; then \
				echo ""; \
				read -p "Enter custom data directory path: " custom_path; \
				if [ -d "$$custom_path" ]; then \
					sed -i 's|^DATA_DIR.*|DATA_DIR      = '"$$custom_path"'|' Makefile; \
					echo "DATA_DIR updated to: $$custom_path"; \
					echo "Updated Makefile:"; \
					grep "^DATA_DIR" Makefile; \
				else \
					echo "Error: Directory '$$custom_path' does not exist!"; \
					exit 1; \
				fi; \
			else \
				echo "Invalid index! Please select between 1 and $$index"; \
				exit 1; \
			fi; \
		else \
			if [ -d "$$choice" ]; then \
				sed -i 's|^DATA_DIR.*|DATA_DIR      = '"$$choice"'|' Makefile; \
				echo "DATA_DIR updated to: $$choice"; \
				echo "Updated Makefile:"; \
				grep "^DATA_DIR" Makefile; \
			else \
				echo "Error: Directory '$$choice' does not exist!"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "No 'data' directory found!"; \
		exit 1; \
	fi
######################################################################
# Save results
save:
	@echo "Saving results..."
	@if [ ! -d "results" ]; then mkdir -p results; fi
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		DATASET_NAME=$$(basename $$(dirname $(DATA_DIR))); \
		echo "Dataset name: $$DATASET_NAME"; \
		if [ -d "results/$$DATASET_NAME" ]; then \
			echo "Results for $$DATASET_NAME already exist. Removing old results..."; \
			rm -rf "results/$$DATASET_NAME"; \
		fi; \
		cp -r $(OUTPUT_DIR) "results/$$DATASET_NAME"; \
		echo "Results saved to: results/$$DATASET_NAME/"; \
		echo "Contents:"; \
		ls -la "results/$$DATASET_NAME/"; \
	else \
		echo "No output directory found. Run 'make cpu' or 'make gpu' first."; \
		exit 1; \
	fi

######################################################################
# Help
help:
	@echo "========================================================="
	@echo "KLT Feature Tracker Makefile"
	@echo "---------------------------------------------------------"
	@echo "Compilation & Profiling:"
	@echo "  make all           - Build libs and run both CPU and GPU"
	@echo "  make lib           - Build default library set"
	@echo "  make lib-cpu       - Build CPU static library (libklt.a)"
	@echo "  make lib-gpu       - Build GPU static library (libklt_gpu.a)"
	@echo "  make cpu           - Build and run CPU version"
	@echo "  make gpu           - Build and run GPU version"
	@echo "  make compare       - Run both CPU and GPU"
	@echo "  make cpu-profile   - Profile CPU version and export report"
	@echo "  make gpu-profile   - Profile GPU version and export report"
	@echo "  make clean         - Clean build and outputs"
	@echo "  make clean-all     - Clean everything (incl. profiles)"
	@echo ""
	@echo "Data Management:"
	@echo "  make path          - List datasets and set DATA_DIR"
	@echo "  make save          - Copy output/ to results/<dataset>/"
	@echo "========================================================="
