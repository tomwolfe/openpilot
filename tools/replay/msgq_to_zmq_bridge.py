#!/usr/bin/env python3
"""
Optional msgq to ZMQ bridge for external tools (plotjuggler, replay UI, etc.).

This bridge subscribes to msgq endpoints and republishes them via ZMQ TCP,
allowing external tools to connect without requiring msgq shared memory access.

Usage:
  # Bridge all services to ZMQ (default port mapping)
  ./msgq_to_zmq_bridge.py

  # Bridge specific services only
  ./msgq_to_zmq_bridge.py --services can,carState,controlsState

  # Bridge with custom ZMQ bind address
  ./msgq_to_zmq_bridge.py --bind-address 0.0.0.0
"""

import argparse
import signal
import sys
import threading
import time

import zmq

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST


class MsgqToZmqBridge:
  """Bridge that forwards msgq messages to ZMQ subscribers."""

  def __init__(self, services: list[str], bind_address: str = "127.0.0.1"):
    self.services = services
    self.bind_address = bind_address
    self.zmq_context = zmq.Context()
    self.pub_sockets: dict[str, zmq.Socket] = {}
    self.running = False
    self.threads: list[threading.Thread] = []

    # Setup signal handlers
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

  def _signal_handler(self, signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    self.running = False

  def _get_port(self, endpoint: str) -> int:
    """Generate a consistent port number from endpoint name."""
    # Use a simple hash-based port mapping similar to the old bridge
    start_port = 8023
    hash_value = hash(endpoint) & 0xFFFFFFFF
    return start_port + (hash_value % (65535 - start_port))

  def _setup_zmq_pubs(self):
    """Create ZMQ publisher sockets for each service."""
    for service in self.services:
      port = self._get_port(service)
      sock = self.zmq_context.socket(zmq.PUB)
      sock.setsockopt(zmq.SNDHWM, 1000)  # High water mark
      bind_addr = f"tcp://{self.bind_address}:{port}"
      try:
        sock.bind(bind_addr)
        self.pub_sockets[service] = sock
        print(f"ZMQ PUB bound: {service} -> {bind_addr}")
      except zmq.ZMQError as e:
        print(f"Warning: Failed to bind {service} to {bind_addr}: {e}")

  def _bridge_service(self, service: str):
    """Bridge a single service from msgq to ZMQ."""
    if service not in self.pub_sockets:
      return

    pub_sock = self.pub_sockets[service]
    sub_sock = messaging.sub_sock(service, conflate=True, timeout=100)

    print(f"Bridge started: {service} (msgq -> ZMQ)")

    while self.running:
      try:
        msg = sub_sock.receive(non_blocking=True)
        if msg is not None:
          try:
            pub_sock.send(msg, flags=zmq.NOBLOCK)
          except zmq.Again:
            # Subscriber is slow, drop message
            pass
      except Exception as e:
        if self.running:
          print(f"Error bridging {service}: {e}")
        break

    # Cleanup
    if service in self.pub_sockets:
      self.pub_sockets[service].close()

  def start(self):
    """Start the bridge."""
    print(f"Starting msgq to ZMQ bridge for services: {', '.join(self.services)}")
    print(f"Binding to: {self.bind_address}")

    self.running = True
    self._setup_zmq_pubs()

    # Start a thread for each service
    for service in self.services:
      if service in self.pub_sockets:
        t = threading.Thread(target=self._bridge_service, args=(service,), daemon=True)
        t.start()
        self.threads.append(t)

    # Main loop - just wait for shutdown
    try:
      while self.running:
        time.sleep(0.1)
    except KeyboardInterrupt:
      pass
    finally:
      self.shutdown()

  def shutdown(self):
    """Shutdown the bridge."""
    print("Shutting down bridge...")
    self.running = False

    # Wait for threads to finish
    for t in self.threads:
      t.join(timeout=1.0)

    # Close ZMQ context
    self.zmq_context.term()
    print("Bridge shutdown complete")


def main():
  parser = argparse.ArgumentParser(description="Bridge msgq messages to ZMQ for external tools")
  parser.add_argument("--services", type=str, default="",
                      help="Comma-separated list of services to bridge (default: all)")
  parser.add_argument("--bind-address", type=str, default="127.0.0.1",
                      help="Address to bind ZMQ sockets to (default: 127.0.0.1)")
  args = parser.parse_args()

  # Get service list
  if args.services:
    services = [s.strip() for s in args.services.split(",") if s.strip() in SERVICE_LIST]
    if not services:
      print("Error: No valid services specified")
      sys.exit(1)
  else:
    # Default: bridge common debugging services
    services = list(SERVICE_LIST.keys())

  bridge = MsgqToZmqBridge(services, args.bind_address)
  bridge.start()


if __name__ == "__main__":
  main()
