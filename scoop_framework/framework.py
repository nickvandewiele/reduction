#!/usr/bin/env python
#
#    This file is part of Scalable COncurrent Operations in Python (SCOOP).
#
#    SCOOP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    SCOOP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with SCOOP. If not, see <http://www.gnu.org/licenses/>.
#
import scoop
scoop.DEBUG = False

import unittest
import subprocess
import time
import copy
import os
import sys
import operator
import signal
import math

from scoop import futures, _control, utils, shared
from scoop._types import FutureQueue
from scoop.broker.structs import BrokerInfo


subprocesses = []
def cleanSubprocesses():
    [a.kill() for a in subprocesses]

try:
    signal.signal(signal.SIGQUIT, cleanSubprocesses)
except AttributeError:
    # SIGQUIT doesn't exist on Windows
    signal.signal(signal.SIGTERM, cleanSubprocesses)


def port_ready(port, socket):
    """Checks if a given port is already binded"""
    try:
        socket.connect(('127.0.0.1', port))
    except IOError:
        return False
    else:
        socket.shutdown(2)
        socket.close()
        return True


class TestScoopCommon(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # Parent initialization
        super(TestScoopCommon, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        global subprocesses
        import socket, datetime, time

        # Start the server
        cls.server = subprocess.Popen([sys.executable, "-m", "scoop.broker.__main__",
        "--tPort", "5555", "--mPort", "5556"])
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        begin = datetime.datetime.now()
        while not port_ready(5555, s):
            if (datetime.datetime.now() - begin > datetime.timedelta(seconds=3)):
                raise Exception('Could not start server!')
        subprocesses.append(cls.server)

        # Setup worker environment
        scoop.IS_RUNNING = True
        scoop.IS_ORIGIN = True
        scoop.WORKER_NAME = 'origin'.encode()
        scoop.BROKER_NAME = 'broker'.encode()
        scoop.BROKER = BrokerInfo("127.0.0.1",
                                  5555,
                                  5556,
                                  "127.0.0.1")
        scoop.worker = (scoop.WORKER_NAME, scoop.BROKER_NAME)
        scoop.MAIN_MODULE = "tests.py"
        scoop.VALID = True
        scoop.DEBUG = False
        scoop.SIZE = 1
        _control.execQueue = FutureQueue()


    @classmethod
    def tearDownClass(cls):
        global subprocesses
        import socket, datetime, time
        _control.execQueue.shutdown()
        del _control.execQueue
        _control.futureDict.clear()
        try:
            cls.w.terminate()
            cls.w.wait()
        except:
            pass
        # Destroy the server
        if cls.server.poll() == None:
            try:
                cls.server.terminate()
                cls.server.wait()
            except:
                pass
        # Stabilise zmq after a deleted socket
        del subprocesses[:]

        # Wait for the previous server to be correctly terminated
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        begin = datetime.datetime.now()
        while port_ready(5555, s):
            if (datetime.datetime.now() - begin > datetime.timedelta(seconds=3)):
                raise Exception('Could not terminate server!')
        s.close()


