CC=gcc
LIBS = -lmero
INC = -I/usr/include/mero -I.
FLAGS = -fPIC -fno-common -Wall -Werror -Wno-attributes \
-fno-strict-aliasing -fno-omit-frame-pointer -D_REENTRANT \
-D_GNU_SOURCE -DM0_INTERNAL='' -DM0_EXTERN=extern -Wall -Werror \
-Wno-attributes -fno-omit-frame-pointer -ggdb -O3 \
-Wno-unused-but-set-variable -rdynamic
FLAGS += $(CFLAGS)
CSOURCES=clovis_functions.c
COBJECTS=$(CSOURCES:%.c=%.o)
TEST_CSOURCES=test_clovis_functions.c
TEST_COBJECTS=$(TEST_CSOURCES:%.c=%.o)
STATICNAME=libclovis.a
SHARENAME=libclovis.so
MODULENAME=pyclovis
LIBDIR=/usr/lib

all: $(MODULENAME)

%.o: %.c
	${CC} $(INC) $(LIBS) $(FLAGS) -c $<

$(STATICNAME): FLAGS+=-static
$(STATICNAME): $(COBJECTS)
	ar rcs $@ $^

static: $(STATICNAME)

$(SHARENAME): $(COBJECTS)
	${CC} $^ -o $@ $(INC) $(LIBS) $(FLAGS) -shared

shared: $(SHARENAME)

$(MODULENAME).so: $(MODULENAME).pyx static
	python setup.py build_ext -i

$(MODULENAME): $(MODULENAME).so

clovis-test: $(TEST_COBJECTS) static
	${CC} $< -o $@ $(INC) $(LIBS) $(FLAGS) -L. -lclovis

test-clovis: clovis-test
	./clovis-test

test-pyclovis: pyclovis.so
	python test_pyclovis.py

test: test-clovis test-pyclovis

clean:
	rm -rf *.so *.a *.o build

install: $(SHARENAME)
	install -m 555 $^ $(LIBDIR)
