# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Handle the "console" type of messages from the client.
"""

import logging

import tornado.ioloop
import tornado.gen
from tornado.escape import json_decode, json_encode

from .base import BaseRequestHandler
from .log import WebSocketLogHandler


logger = logging.getLogger(__name__)


class ConsoleAJAXHandler(BaseRequestHandler):
    """
    Handle the AJAX requests from the client to control the tasks.
    """
    # Allow only one task running at the same time
    onetask_only = True

    def initialize(self):
        """Hook for subclass initialization.  Called for each request."""
        self.configs = self.application.configmanager
        self.status = self.application.task_status
        # NOTE:
        # Use ``IOLoop.instance`` instead of ``IOLoop.current``, since we
        # will need to communicate with the main thread (e.g., callback)
        # from another thread, which executes the submitted task.
        self.io_loop = tornado.ioloop.IOLoop.instance()

    @tornado.web.authenticated
    def get(self):
        """
        Handle the READ-ONLY tasks operations.

        Supported actions:
        - get: Get the current status of tasks.
        """
        action = self.get_argument("action", "get")
        if action == "get":
            response = {"status": self.status}
            success = True
        else:
            # ERROR: bad action
            success = False
            reason = "Bad request action: {0}".format(action)
        #
        if success:
            logger.debug("Response: {0}".format(response))
            self.set_header("Content-Type", "application/json; charset=UTF-8")
            self.write(json_encode(response))
        else:
            logger.warning("Request failed: {0}".format(reason))
            self.send_error(400, reason=reason)

    @tornado.web.authenticated
    def post(self):
        """
        Handle the READ-WRITE task operations.

        XXX/TODO:
        * How to kill the submitted task? (force kill thread?)

        Supported actions:
        - start: Start the default or specified task.
        - stop: Stop the running task (TODO/XXX)
        """
        request = json_decode(self.request.body)
        logger.debug("Received request: {0}".format(request))
        action = request.get("action")
        if action == "start":
            task = request.get("task")
            kwargs = request.get("kwargs", {})
            success, reason = self._start_task(task=task, kwargs=kwargs)
        elif action == "stop":
            success = False
            reason = "NOT implemented action: {0}".format(action)
        else:
            # ERROR: bad action
            success = False
            reason = "Bad request action: {0}".format(action)
        #
        if success:
            response = {"status": self.status}
            logger.debug("Response: {0}".format(response))
            self.set_header("Content-Type", "application/json; charset=UTF-8")
            self.write(json_encode(response))
        else:
            logger.warning("Request failed: {0}".format(reason))
            self.send_error(400, reason=reason)

    # FIXME/XXX:
    # * How to call this task asynchronously ??
    def _start_task(self, task=None, kwargs={}):
        """
        Start the task by submitting it to the executor.

        Parameters
        ----------
        task : str, optional
            The name of the task to be started.
            If not specified, then start the default task.
            NOTE:
            Currently only support the default (``None``) and ``test`` tasks.
        kwargs : dict, optional
            Keyword arguments to be passed to the task.

        Returns
        -------
        success : bool
            Whether the task successfully finished?
        error : str
            Error message if the task failed
        """
        TASKS = {
            None: self._task_default,  # default task
            "test": self._task_test,
        }
        try:
            f_task = TASKS[task]
        except KeyError:
            success = False
            error = "Unknown task: {0}".format(task)
            logger.warning(error)
            return (success, error)
        #
        if self.onetask_only and self.status["running"]:
            success = False
            error = "Task already running and only one task allowed"
            logger.warning(error)
        else:
            logger.info("Submit the task to the executor ...")
            self.status["running"] = True
            self.status["finished"] = False
            # Also push the logging messages to clients through WebSocket
            self._add_wsloghandler()
            future = self.application.executor.submit(f_task, **kwargs)
            self.io_loop.add_future(future, self._task_callback)
            success, error = future.result()
        return (success, error)

    def _task_default(self, **kwargs):
        """
        The default task that this console manages, which performs
        the foregrounds simulations.

        Returns
        -------
        success : bool
            Whether the task successfully finished?
        error : str
            Error message if the task failed

        NOTE
        ----
        The task is synchronous and may be computationally intensive
        (i.e., CPU-bound rather than IO/event-bound), therefore,
        threads (or processes) are required to make it non-blocking
        (i.e., asynchronous).

        References:
        [1] https://stackoverflow.com/a/32164711/4856091
        """
        logger.info("Console DEFAULT task: START ...")
        logger.info("Preparing to start foregrounds simulations ...")
        logger.info("Importing modules + Numba JIT, waiting ...")
        from ..foregrounds import Foregrounds
        #
        logger.info("Checking the configurations ...")
        self.configs.check_all()
        fg = Foregrounds(self.configs)
        fg.preprocess()
        fg.simulate()
        fg.postprocess()
        logger.info("Foregrounds simulations DONE!")
        logger.info("Console DEFAULT task: DONE!")
        # NOTE: always return a tuple of (success, error)
        return (True, None)

    def _task_test(self, **kwargs):
        """
        Test task ...
        """
        import time
        logger.info("Console TEST task: START ...")
        for i in range(kwargs["time"]):
            logger.info("Console TEST task: slept {0} seconds ...".format(i))
            time.sleep(1)
        logger.info("Console TEST task: DONE!")
        return (True, None)

    def _task_callback(self, future):
        """Callback function executed when the task finishes"""
        logger.info("Task finished! Callback ...")
        self.status["running"] = False
        self.status["finished"] = True
        self._remove_wsloghandler()
        logger.info("Callback DONE!")

    def _add_wsloghandler(self):
        """
        Add a WebSocket handler to the logging handlers, which will capture
        all the logging messages and push them to the client.
        """
        self.wsloghandler = WebSocketLogHandler(self.application.websockets,
                                                msg_type="console")
        root_logger = logging.getLogger()
        root_logger.addHandler(self.wsloghandler)
        logger.info("Added the WebSocket logging handler")

    def _remove_wsloghandler(self):
        """Remove the WebSocket logging handler"""
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.wsloghandler)
        logger.info("Removed the WebSocket logging handler")
