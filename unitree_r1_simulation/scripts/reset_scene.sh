#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

case "${1:-objects}" in
    objects) code=1 ;;
    all)     code=2 ;;
    *)       die "usage: $0 [objects|all]" ;;
esac
compose run --rm -T --entrypoint python teleop -c "
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
ChannelFactoryInitialize(1, '${DDS_INTERFACE:-lo}')
pub = ChannelPublisher('rt/reset_pose/cmd', String_)
pub.Init()
time.sleep(1.0)
pub.Write(String_(data='${code}'))
time.sleep(0.5)
" >/dev/null
info "reset request '${code}' sent (the simulator log shows 'objects reset')"
