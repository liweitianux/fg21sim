# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Handle the "console" type of messages from the client.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.gen

from .loghandler import WebSocketLogHandler


logger = logging.getLogger(__name__)


class ConsoleHandler:
    """
    Handle the "console" type of messages from the client.

    XXX/TODO:
    * How to kill the submitted task? (force kill thread?)

    Parameters
    ----------
    websocket : `~tornado.websocket.WebSocketHandler`
        An `~tornado.websocket.WebSocketHandler` instance.
        The ``WebSocketLogHandler`` requires this to push logging messages
        to the client.
    max_workers : int, optional
        Maximum number of workers/threads for the execution pool
    onetask_only : bool, optional
        Whether to allow only one task running at the same time?

    Attributes
    ----------
    websocket : `~tornado.websocket.WebSocketHandler`
        An `~tornado.websocket.WebSocketHandler` instance, which is used by
        the ``WebSocketLogHandler`` to push logging messages to the client.
    wsloghandler : `~WebSocketLogHandler`
        Push logging messages to the client through WebSocket.
    executor : `~concurrent.futures.ThreadPoolExecutor`
        Where to submit the synchronous task and make it perform
        asynchronously using threads.
    io_loop : `~tornado.ioloop.IOLoop`
        Used to communicate with the main thread (e.g., callback) from the
        submitted task, which is executed on a different thread.
    onetask_only : bool
        Whether to allow only one task running at the same time?
    status : dict
        Whether the task is running and/or finished?
        There may be 4 possible status:
        1. running=False, finished=False: not started
        2. running=False, finished=True:  finished
        3. running=True,  finished=False: running
        4. running=True,  finished=True:  ?? error ??
    """
    def __init__(self, websocket, max_workers=3, onetask_only=False):
        self.websocket = websocket
        self.wsloghandler = WebSocketLogHandler(websocket, msg_type="console")
        self.executor = ThreadPoolExecutor(max_workers=1)
        # NOTE:
        # Use ``IOLoop.instance`` instead of ``IOLoop.current``, since we
        # will need to communicate with the main thread from another thread.
        self.io_loop = tornado.ioloop.IOLoop.instance()
        self.onetask_only = onetask_only
        self.status = {"running": False, "finished": False}

    def handle_message(self, msg):
        try:
            msg_type = msg["type"]
            msg_action = msg["action"]
            response = {"type": msg_type, "action": msg_action}
            logger.info("WebSocket: handle message: " +
                        "type: {0}, action: {1}".format(msg_type, msg_action))
            if msg_action == "start":
                # FIXME/XXX: This task should be asynchronous!
                success, error = self._start()
                response["success"] = success
                if not success:
                    response["error"] = error
            elif msg_action == "start_test":
                # FIXME/XXX: This task should be asynchronous!
                success, error = self._start(msg["time"])
                response["success"] = success
                if not success:
                    response["error"] = error
            elif msg_action == "get_status":
                response["success"] = True
                response["action"] = "push"
                response["status"] = self.status
            else:
                logger.warning("WebSocket: " +
                               "unknown action: {0}".format(msg_action))
                response["success"] = False
                response["error"] = "unknown action: {0}".format(msg_action)
        except KeyError:
            # Received message has wrong syntax/format
            response = {"success": False,
                        "type": msg_type,
                        "error": "no action specified"}
        #
        logger.debug("WebSocket: response: {0}".format(response))
        return response

    # FIXME/XXX:
    # * How to call this task asynchronously ??
    def _start_test(self, *args, **kwargs):
        """
        Start the task by submitting it to the executor

        Returns
        -------
        success : bool
            Whether success without any errors
        error : str
            Detail of the error if not succeed

        """
        if self.onetask_only and self.status["running"]:
            logger.warning("Task already running, and only one task allowed")
            success = False
            error = "already running and only one task allowed"
        else:
            logger.info("Start the task on the executor ...")
            self.status["running"] = True
            self.status["finished"] = False
            # Also push the logging messages to the client
            self._add_wsloghandler()
            future = self.executor.submit(self._task_test, *args, **kwargs)
            self.io_loop.add_future(future, self._task_callback)
            success, error = future.result()
        return (success, error)

    # FIXME/XXX:
    # * How to call this task asynchronously ??
    def _start(self, *args, **kwargs):
        """
        Start the task by submitting it to the executor

        Returns
        -------
        success : bool
            Whether success without any errors
        error : str
            Detail of the error if not succeed

        """
        if self.onetask_only and self.status["running"]:
            logger.warning("Task already running, and only one task allowed")
            success = False
            error = "already running and only one task allowed"
        else:
            logger.info("Start the task on the executor ...")
            self.status["running"] = True
            self.status["finished"] = False
            # Also push the logging messages to the client
            self._add_wsloghandler()
            future = self.executor.submit(self._task, *args, **kwargs)
            self.io_loop.add_future(future, self._task_callback)
            success, error = future.result()
        return (success, error)

    def _response_future(self, future):
        """
        Callback function which will be called when the caller finishes
        in order to response the results to the client.
        """
        response = {"type": "console", "action": "future"}
        success, error = future.result()
        response["success"] = success
        if not success:
            response["error"] = error
        logger.debug("WebSocket: future response: {0}".format(response))
        msg_response = json.dumps(response)
        self.websocket.write_message(msg_response)

    def _add_wsloghandler(self):
        """Add the ``self.wsloghandler`` to the logging handlers"""
        root_logger = logging.getLogger()
        root_logger.addHandler(self.wsloghandler)
        logger.info("Added the WebSocket logging handler")

    def _remove_wsloghandler(self):
        """Remove the ``self.wsloghandler`` from the logging handlers"""
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.wsloghandler)
        logger.info("Removed the WebSocket logging handler")

    def _task_callback(self, future):
        """Callback function executed when the task finishes"""
        logger.info("Console task finished! Callback ...")
        self.status["running"] = False
        self.status["finished"] = True
        self._remove_wsloghandler()
        #
        response = {"type": "console", "action": "push"}
        response["success"] = True
        response["status"] = self.status
        logger.debug("WebSocket: future response: {0}".format(response))
        msg_response = json.dumps(response)
        self.websocket.write_message(msg_response)

    def _task_test(self, *args, **kwargs):
        """
        The task this console to manage.

        Returns
        -------
        success : bool
            Whether success without any errors
        error : str
            Detail of the error if not succeed

        NOTE
        ----
        The task is synchronous and may be computationally intensive
        (i.e., CPU-bound rather than IO/event-bound), therefore,
        threads (or processes) are required to make it non-blocking
        (i.e., asynchronous).

        Credit: https://stackoverflow.com/a/32164711/4856091
        """
        import time
        logger.info("console task: START")
        for i in range(args[0]):
            logger.info("console task: slept {0} seconds ...".format(i))
            time.sleep(1)
        logger.info("console task: DONE!")
        return (True, None)

    def _task(self, *args, **kwargs):
        """
        The task this console to manage.
        Perform the foregrounds simulations.

        Returns
        -------
        success : bool
            Whether success without any errors
        error : str
            Detail of the error if not succeed

        NOTE
        ----
        The task is synchronous and may be computationally intensive
        (i.e., CPU-bound rather than IO/event-bound), therefore,
        threads (or processes) are required to make it non-blocking
        (i.e., asynchronous).

        Credit: https://stackoverflow.com/a/32164711/4856091
        """
        logger.info("Preparing to start foregrounds simulations ...")
        logger.info("Importing modules + Numba JIT, waiting ...")

        from ..foregrounds import Foregrounds

        # FIXME: This is a hack
        configs = self.websocket.configs
        logger.info("Checking the configurations ...")
        configs.check_all()

        fg = Foregrounds(configs)
        fg.preprocess()
        fg.simulate()
        fg.postprocess()

        logger.info("Foregrounds simulations DONE!")

        # NOTE: Should always return a tuple of (success, error)
        return (True, None)
