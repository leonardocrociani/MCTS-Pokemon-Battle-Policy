#!/bin/bash

# useful to kill the engine if needed
kill $(ps xa | grep "pokemon" | sort | head -1 | awk '{ print $1 }')