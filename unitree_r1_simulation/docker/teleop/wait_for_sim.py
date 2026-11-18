#!/usr/bin/env python3

import argparse
import sys
import threading
import time

import zmq


def request_camera_config(host: str, port: int, timeout_ms: int = 1000):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    try:
        sock.send(b"GET_DATA")
        if sock.poll(timeout_ms):
            return sock.recv_json()
        return None
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Block until the simulator is ready for xr_teleoperate.')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config-port", type=int, default=60000)
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds (first Isaac Sim start is slow)")
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout

    print(f"[wait_for_sim] waiting for the R1 simulator camera server at {args.host}:{args.config_port} ...",
          flush=True)
    config = None
    last_note = 0.0
    while config is None:
        if time.monotonic() > deadline:
            print("[wait_for_sim] timed out - is the r1-sim container running?", flush=True)
            return 1
        config = request_camera_config(args.host, args.config_port)
        if config is None:
            if time.monotonic() - last_note > 15.0:
                print("[wait_for_sim]   ... still waiting (Isaac Sim can take several minutes on first start)",
                      flush=True)
                last_note = time.monotonic()
            time.sleep(1.0)
    head = config["head_camera"]
    print(f"[wait_for_sim] camera config OK: {head['image_shape']} binocular={head['binocular']}", flush=True)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    got = threading.Event()
    ChannelFactoryInitialize(args.domain, args.interface or None)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda msg: got.set(), 1)
    print(f"[wait_for_sim] waiting for rt/lowstate (DDS domain {args.domain}, interface {args.interface}) ...",
          flush=True)
    while not got.wait(1.0):
        if time.monotonic() > deadline:
            print("[wait_for_sim] no rt/lowstate received - check DDS_INTERFACE / DDS_DOMAIN_ID", flush=True)
            return 1
    print("[wait_for_sim] simulator is up", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
