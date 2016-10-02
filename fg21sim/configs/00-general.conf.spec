# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the general configurations, which control the general
# behaviors, or will be used in other configuration sections.


# Configurations for initialization/reconfiguration of the `logging` module
[logging]
# DEBUG:    Detailed information, typically of interest only when diagnosing
#           problems.
# INFO:     Confirmation that things are working as expected.
# WARNING:  An dinciation that something unexpected happended, or indicative
#           of some problem in the near future (e.g., "disk space low").
#           The software is still working as expected.
# ERROR:    Due to a more serious problem, the software has not been able to
#           perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable
#           to continue running.
level = option("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", default="INFO")

# Set the format of displayed messages
format = string(default="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Set the date/time format in messages
datefmt = string(default="%Y-%m-%dT%H:%M:%S")

# Set the logging filename (will create a `FileHandler`)
# If set to "" (empty string), then the `FileHandler` will be disabled.
filename = string(default="")
# Set the mode to open the above logging file (default: "append" mode)
filemode = option("w", "a", default="a")

# Set the stream used to initialize the `StreamHandler`
# If set to "" (empty string), then the `StreamHandler` will be disabled.
stream = option("stderr", "stdout", "", default="stderr")
