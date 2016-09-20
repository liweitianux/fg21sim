# Makefile for "fg21sim"
#
# 2016-09-20
#

init:
	pip3 install --user -r requirements.txt

test:
	nosetests tests
