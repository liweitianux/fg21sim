# Makefile for "fg21sim"

default:
	@echo "+-------------------------------------------------------------+"
	@echo "|                 Make Utility for fg21sim                    |"
	@echo "+-------------------------------------------------------------+"
	@echo "Available targets:"
	@echo "  + venv"
	@echo "        create virtualenv 'venv' and install the dependencies"
	@echo "  + test"
	@echo "        run the test cases"

# Create virtualenv and install/update the dependencies
venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d "venv" || virtualenv -p python3 venv
	./venv/bin/pip3 install --user -r requirements.txt
	touch venv/bin/activate

# Install this package to the virtualenv
devbuild: venv
	./venv/bin/python3 setup.py install

# Run the test cases
test: devbuild
	./venv/bin/python3 tests/runtests.py
