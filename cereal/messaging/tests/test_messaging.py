import capnp
import multiprocessing
import numbers
import os
import random
import threading
import time
from openpilot.common.parameterized import parameterized
import pytest

from cereal import log, car
import cereal.messaging as messaging
from cereal.services import SERVICE_LIST

events = [evt for evt in log.Event.schema.union_fields if evt in SERVICE_LIST.keys()]

def random_sock():
  return random.choice(events)

def random_socks(num_socks=10):
  return list({random_sock() for _ in range(num_socks)})

def random_bytes(length=1000):
  return bytes([random.randrange(0xFF) for _ in range(length)])

def zmq_sleep(t=1):
  if "ZMQ" in os.environ:
    time.sleep(t)


# TODO: this should take any capnp struct and returrn a msg with random populated data
def random_carstate():
  fields = ["vEgo", "aEgo", "brake", "steeringAngleDeg"]
  msg = messaging.new_message("carState")
  cs = msg.carState
  for f in fields:
    setattr(cs, f, random.random() * 10)
  return msg

# TODO: this should compare any capnp structs
def assert_carstate(cs1, cs2):
  for f in car.CarState.schema.non_union_fields:
    # TODO: check all types
    val1, val2 = getattr(cs1, f), getattr(cs2, f)
    if isinstance(val1, numbers.Number):
      assert val1 == val2, f"{f}: sent '{val1}' vs recvd '{val2}'"

def delayed_send(delay, sock, dat):
  def send_func():
    sock.send(dat)
  threading.Timer(delay, send_func).start()


class TestMessaging:
  def setUp(self):
    # msgq pub socket takes too long to die
    # sleep to prevent multiple publishers error between tests
    time.sleep(0.1)

  @parameterized.expand(events)
  def test_new_message(self, evt):
    try:
      msg = messaging.new_message(evt)
    except capnp.lib.capnp.KjException:
      msg = messaging.new_message(evt, random.randrange(200))
    assert (time.monotonic() - msg.logMonoTime) < 0.1
    assert not msg.valid
    assert evt == msg.which()

  @parameterized.expand(events)
  def test_pub_sock(self, evt):
    messaging.pub_sock(evt)

  @parameterized.expand(events)
  def test_sub_sock(self, evt):
    messaging.sub_sock(evt)

  @parameterized.expand([
    (messaging.drain_sock, capnp._DynamicStructReader),
    (messaging.drain_sock_raw, bytes),
  ])
  def test_drain_sock(self, func, expected_type):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=1000)

    # no wait and no msgs in queue
    msgs = func(sub_sock)
    assert isinstance(msgs, list)
    assert len(msgs) == 0

    # no wait but msgs are queued up
    num_msgs = random.randrange(3, 10)
    for _ in range(num_msgs):
      pub_sock.send(messaging.new_message(sock).to_bytes())
    time.sleep(0.1)
    msgs = func(sub_sock)
    assert isinstance(msgs, list)
    assert all(isinstance(msg, expected_type) for msg in msgs)
    assert len(msgs) == num_msgs

  def test_recv_sock(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=100)

    # no wait and no msg in queue, socket should timeout
    recvd = messaging.recv_sock(sub_sock)
    assert recvd is None

    # no wait and one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    time.sleep(0.01)
    recvd = messaging.recv_sock(sub_sock)
    assert isinstance(recvd, capnp._DynamicStructReader)
    # https://github.com/python/mypy/issues/13038
    assert_carstate(msg.carState, recvd.carState)

  def test_recv_one(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=1000)

    # no msg in queue, socket should timeout
    recvd = messaging.recv_one(sub_sock)
    assert recvd is None

    # one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    recvd = messaging.recv_one(sub_sock)
    assert isinstance(recvd, capnp._DynamicStructReader)
    assert_carstate(msg.carState, recvd.carState)

  def test_recv_one_or_none(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock)

    # no msg in queue, socket shouldn't block
    recvd = messaging.recv_one_or_none(sub_sock)
    assert recvd is None

    # one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    recvd = messaging.recv_one_or_none(sub_sock)
    assert isinstance(recvd, capnp._DynamicStructReader)
    assert_carstate(msg.carState, recvd.carState)

  def test_recv_one_retry(self):
    sock = "carState"
    sock_timeout = 0.1
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=round(sock_timeout*1000))

    # Test that recv_one_retry blocks until a message is received
    # We use threading instead of multiprocessing due to socket pickling limitations
    result = {"received": False, "msg": None}
    
    def receive_thread():
      msg = messaging.recv_one_retry(sub_sock)
      result["received"] = True
      result["msg"] = msg
    
    # Start thread and wait to verify it's blocking
    t = threading.Thread(target=receive_thread)
    t.start()
    time.sleep(sock_timeout * 3)
    
    # Thread should still be waiting (not received yet)
    assert not result["received"]
    
    # Now send a message
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    
    # Wait for thread to complete
    t.join(timeout=1.0)
    assert result["received"]
    assert isinstance(result["msg"], capnp._DynamicStructReader)
    assert_carstate(msg.carState, result["msg"].carState)
