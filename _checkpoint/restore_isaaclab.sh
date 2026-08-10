#!/bin/bash

set -e

echo "Restoring Isaac Lab 2.3.2 launcher..."

cp "$(dirname "$0")/isaaclab.sh.v2.3.2" /opt/isaaclab-seed/isaaclab.sh
chmod +x /opt/isaaclab-seed/isaaclab.sh

echo "Checking launcher..."
ls -lh /opt/isaaclab-seed/isaaclab.sh

echo "Testing Isaac Lab..."
cd /opt/isaaclab-seed
./isaaclab.sh -p -c "print('ISAACLAB RESTORED OK')"
