PYTHONEXE=python3
CYTHON=cython
NAME=DTW_CCore

all: $(NAME)

$(NAME): $(NAME).pyx
	$(PYTHONEXE) setup.py build_ext --inplace

python2: $(NAME).pyx
	python2 setup.py build_ext --inplace

doc: $(NAME).pyx
	$(CYTHON) -a $(NAME).pyx

clean:
	rm -rf $(NAME).c build

mrproper: clean
	rm -f $(NAME)*.so $(NAME).html

