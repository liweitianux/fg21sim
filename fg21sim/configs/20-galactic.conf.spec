# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the point source configurations, which control the 
# amount of point sources of each types, and the output staffs. 

[extragalactic]

# Extragalactic point sources
[[pointsources]]
# Whether save this point source catelogue to disk
save = boolean(default=True)
# Output directory to save the simulated catelogues
output_dir = string(default="PS_tables")
# PS components to be simulated
pscomponents=string_list(default=list())
# Number of each type of point source
# Star forming
[[[starforming]]]
# Number of samples
numps = integer(default=1000)
# Prefix
prefix = string(default="SF")

[[[starbursting]]]
# Number of samples
numps = integer(default=1000)
# Prefix
prefix = string(default="SB")

[[[radioquiet]]]
# Number of samples
numps = integer(default=1000)
# Prefix
prefix = string(default="RQ")

[[[FRI]]]
# Number of samples
numps = integer(default=1000)
# Prefix
prefix = string(default="FRI")

[[[FRII]]]
# Number of samples
numps = integer(default=1000)
# Prefix
prefix = string(default="FRII")
