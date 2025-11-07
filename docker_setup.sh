#!/bin/bash
apt-get update -y
apt-get install -y --no-install-recommends libglib2.0-0 libgl1
rm -rf /var/lib/apt/lists/*

