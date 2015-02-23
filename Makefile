CC = gcc
CLC = openclc
CLC_ARCH = gpu_32
CFLAGS = -Wall -D_GNU_SOURCE -DHAVE_INLINE -fPIC -std=c99
CLFLAGS = -emit-llvm -c -arch $(CLC_ARCH) -DFLOAT_GPU_UNIT
DEBUG_CFLAGS = -O -ggdb -DSTABLE_MIN_LOG=0
BENCHMARK_CFLAGS =  $(RELEASE_CFLAGS) -DBENCHMARK
RELEASE_CFLAGS = -O3 -march=native -DSTABLE_MIN_LOG=1
PROFILE_CFLAGS = $(RELEASE_CFLAGS) -pg -static-libgcc
E_LIBS = $(shell pkg-config --libs gsl) -pthread

PROJECT_NAME = libstable

CONFS = debug release benchmark
DEFAULT_CONF = debug

OBJDIR = obj
SRCDIR = src
INCDIR = includes
BINDIR = bin
DOCDIR = doc
LIBDIR = lib
LIBSRCDIR = srclib
CLDIR = opencl
TARGETS = example fittest stable_array \
			stable_test stable_performance stable_precision \
			gpu_tests gpu_performance opencl_tests fitperf \
			gpu_mpoints_perftest stable_plot
INCLUDES = -I./includes/

INCS := $(wildcard $(INCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(patsubst %.c,%.o, $(SRCS))

LIB_SRCS := $(wildcard $(LIBSRCDIR)/**/*.c)
LIB_OBJS := $(addprefix $(OBJDIR)/, $(patsubst %.c,%.o, $(LIB_SRCS)))
LIB_NAMES := $(patsubst $(LIBSRCDIR)/%, %, $(wildcard $(LIBSRCDIR)/*))
LIBS :=	$(addsuffix .a, $(LIB_NAMES))
LIBS +=	$(addsuffix .so, $(LIB_NAMES))
LIB_DEPS := $(addsuffix .deps, $(addprefix $(OBJDIR)/., $(LIB_NAMES)))
LIB_OBJDIRS := $(foreach conf, $(CONFS), $(addprefix $(OBJDIR)/$(conf)/$(LIBSRCDIR)/, $(LIB_NAMES)))
LIB_OUTDIRS := $(addprefix $(LIBDIR)/, $(CONFS))

CL_SRCS := $(wildcard $(CLDIR)/*.cl)
CL_OBJS := $(patsubst $(CLDIR)/%.cl, $(OBJDIR)/%.bc, $(CL_SRCS))

TEST_SRCS := $(wildcard $(TESTDIR)/*.c)
TEST_OBJS := $(patsubst %.c,%.o, $(TEST_SRCS))
OBJS_NOMAIN := $(filter-out $(addprefix %/, $(addsuffix .o, $(TARGETS))), $(OBJS))

DOC_TEXS := $(wildcard $(DOCDIR)/*.tex)
DOC_TEXS += $(DOCDIR)/latex/refman.tex
DOC_PDFS := $(patsubst %.tex, %.pdf, $(DOC_TEXS))

DOXY_OUTPUTS := $(filter-out $(DOCDIR)/, $(dir $(wildcard $(DOCDIR)/*/)))

FMT_BOLD := $(shell tput bold)
FMT_NORM := $(shell tput sgr0)

TAR_EXCLUDES = bin obj doc .tar.gz .git tasks \
		cscope.out $(PROJECT_NAME).sublime-project $(PROJECT_NAME).sublime-workspace \
		*.dat callgrind.* gmon.out
TAR_EXCLUDES_ARG = $(addprefix --exclude=, $(TAR_EXCLUDES))

include Makefile.$(shell uname)

### Makefile plugins

### End Makefile plugins

.PRECIOUS: %.o %.d %.g
.PHONY: benchmark clean pack doxydoc docclean benchmark-run configs $(TARGETS) depend

all: $(CONFS)
libs: $(LIBS)
final: all docs pack

### Compilation

$(CONFS): configs

flags.dat: Makefile
	@echo $(CFLAGS) $(BENCH_CFLAGS) > $@

scan: clean
	@scan-build make debug

$(OBJS): | $(OBJDIR)

## Dependencies

depend: $(OBJDIR)/.deps $(LIB_DEPS)

$(OBJDIR)/.deps: $(SRCS) Makefile | $(OBJDIR)
	@-rm -f $(OBJDIR)/.deps
	@$(CC) $(CFLAGS) $(INCLUDES) -MM $(filter-out Makefile, $^) >> $(OBJDIR)/.deps;
	@awk '{if (sub(/\\$$/,"")) printf "%s", $$0; else print $$0}' $@ > "$@.0"
	@mv "$@.0" $@
	@for c in $(CONFS); do \
		awk '{printf("$(OBJDIR)/%s/$(SRCDIR)/%s\n", conf, $$0)}' conf=$$c $@ >> "$@.0"; \
	done
	@mv "$@.0" $@

# Library dependency detection.
$(OBJDIR)/.%.deps: $(LIBSRCDIR)/**/*.c Makefile | $(OBJDIR)
	@-rm -f $@
	@$(CC) $(CFLAGS) $(INCLUDES) -MM $(filter-out Makefile, $^) >> $@;
	@awk '{if (sub(/\\$$/,"")) printf "%s", $$0; else print $$0}' $@ > "$@.0"
	@mv "$@.0" $@
	@for c in $(CONFS); do \
		awk '{printf("$(OBJDIR)/%s/$(LIBSRCDIR)/$*/%s | $(LIBDIR)\n", conf, $$0)}' conf=$$c $@ >> "$@.0"; \
		echo "$(LIBDIR)/$$c/$*.a: $(patsubst %.c, $(OBJDIR)/$$c/%.o, $(filter-out Makefile, $^))" >> "$@.0"; \
		echo "$(LIBDIR)/$$c/$*.so: $(patsubst %.c, $(OBJDIR)/$$c/%.o, $(filter-out Makefile, $^))" >> "$@.0"; \
	done
	@mv "$@.0" $@
	@echo "$*.a: $(LIBDIR)/$(DEFAULT_CONF)/$*.a" >> $@
	@echo "$*.so: $(LIBDIR)/$(DEFAULT_CONF)/$*.so" >> $@

configs: $(OBJDIR)/.conf_flags.mk

$(OBJDIR)/.conf_flags.mk: Makefile | $(OBJDIR)
	@echo "$(FMT_BOLD)Generating configurations...$(FMT_NORM)"
	@-rm -f $@
	@for c in $(CONFS); do \
		cnf=$$(echo $$c | tr '[:lower:]' '[:upper:]'); \
		for t in $(TARGETS); do \
			echo "$(BINDIR)/$$c/$$t: CFLAGS += \$$(""$$cnf""_CFLAGS)" >> $@; \
			echo "$(BINDIR)/$$c/$$t: LDFLAGS += \$$(""$$cnf""_LDFLAGS)" >> $@; \
			echo "$(BINDIR)/$$c/$$t: \$$(addprefix $(OBJDIR)/$$c/, \$$(OBJS_NOMAIN)) $(OBJDIR)/$$c/$(SRCDIR)/$$t.o $(addprefix $(LIBDIR)/$$c/, $(LIBS))" >> $@; \
		done; \
		echo "$$c: $(addprefix $(BINDIR)/$$c/, $(TARGETS))" >> $@; \
	done
	@for t in $(TARGETS); do \
		echo "$$t: $(BINDIR)/$(DEFAULT_CONF)/$$t \$$(CL_OBJS)" >> $@; \
	done


-include $(OBJDIR)/.conf_flags.mk
-include $(OBJDIR)/.deps
-include $(LIB_DEPS)

## Directories

$(OBJDIR): Makefile
	@for c in $(CONFS); do \
		echo "$(FMT_BOLD)Creating object directories for configuration $$c...$(FMT_NORM)"; \
		mkdir -p $(OBJDIR)/$$c/$(SRCDIR); \
		mkdir -p $(OBJDIR)/$$c/$(LIBSRCDIR); \
		mkdir -p $(OBJDIR)/$$c/$(TESTDIR); \
	done

$(PCAP_DIR):
	@mkdir -p $@

$(BINDIR): Makefile
	@echo "$(FMT_BOLD)Creating bin directory$(FMT_NORM)"
	@for c in $(CONFS); do \
		mkdir -p $(BINDIR)/$$c; \
	done

$(LIBDIR): Makefile
	@echo "$(FMT_BOLD)Creating lib directory$(FMT_NORM)"
	@mkdir -p $(LIB_OUTDIRS)
	@mkdir -p $(LIB_OBJDIRS)

## Cleaning

clean: codeclean resultclean
	@echo "$(FMT_BOLD)Directory clean.$(FMT_NORM)"

codeclean:
	@echo "$(FMT_BOLD)Cleaning build folders...$(FMT_NORM)"
	@-rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR)

resultclean:
	@echo "$(FMT_BOLD)Removing result...$(FMT_NORM)"
	@-rm -rf *.dat *.g *.gdat *.png

## Libraries

$(LIBDIR)/%.a: | $(LIBDIR)
	@echo "$(FMT_BOLD)Building library $(notdir $@)... $(FMT_NORM)"
	@$(AR) -rv $@ $^

$(LIBDIR)/%.so: | $(LIBDIR)
	@echo "$(FMT_BOLD)Building library $(notdir $@)... $(FMT_NORM)"
	@$(CC) $(CFLAGS) -shared $^ $(E_LIBS) -o $@

## Compilation

$(OBJDIR)/%.o: | $(OBJDIR) depend configs
	@echo "$< -> $@"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

## OpenCL compilation
$(OBJDIR)/%.bc: $(CLDIR)/%.cl Makefile
	@echo "$< - > $@"
	@$(CLC) $(CLFLAGS) $< -o $@

## Executable

$(BINDIR)/%: | $(BINDIR) depend configs
	@echo "$(FMT_BOLD)Building final target: $* $(FMT_NORM)"
	@$(CC) $(CFLAGS) $(INCLUDES) $^ $(E_LIBS) -o $@

## Packing

pack: $(DOC_PDFS) codeclean
	@cd ..; tar $(TAR_EXCLUDES_ARG) -czf $(PROJECT_NAME).tar.gz $(lastword $(notdir $(CURDIR)))
	@echo "Packed $(PROJECT_NAME).tar.gz in parent directory."

