/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * Web UI of "fg21sim"
 * Console operations using the WebSocket communications
 */

"use strict";


/**
 * Update the task status "#task-status" on the page.
 *
 * @param {Object} status - The status pushed from the server is an object
 *                          containing the "running" and "finished" items.
 */
var updateTaskStatus = function (status) {
  var running = status.running;
  var finished = status.finished;
  var ts = null;
  if (!running && !finished) {
    ts = "Not started";
    $("#task-status").removeClass("label-success label-warning label-danger")
      .addClass("label-default");
    $("#task-status .icon").removeClass("fa-check-circle fa-question-circle")
      .removeClass("fa-spin fa-spinner")
      .addClass("fa-coffee");
  }
  else if (!running && finished) {
    ts = "Finished";
    $("#task-status").removeClass("label-default label-warning label-danger")
      .addClass("label-success");
    $("#task-status .icon").removeClass("fa-coffee fa-question-circle")
      .removeClass("fa-spin fa-spinner")
      .addClass("fa-check-circle");
  }
  else if (running && !finished) {
    ts = "Running";
    $("#task-status").removeClass("label-default label-success label-danger")
      .addClass("label-warning");
    $("#task-status .icon").removeClass("fa-coffee fa-check-circle")
      .removeClass("fa-question-circle")
      .addClass("fa-spin fa-spinner");
  }
  else {
    // Unknown status: ERROR ??
    ts = "ERROR?";
    $("#task-status").removeClass("label-default label-success label-warning")
      .addClass("label-danger");
    $("#task-status .icon").removeClass("fa-coffee fa-check-circle")
      .removeClass("fa-spin fa-spinner")
      .addClass("fa-question-circle");
  }
  console.log("Task status:", ts);
  $("#task-status .text").text(ts);
};


/**
 * Get the task status from the server
 *
 * @param {Object} ws - The opened WebSocket object through which to send
 *                      the request.
 */
var getServerTaskStatus = function (ws) {
  var msg = {type: "console", action: "get_status"};
  ws.send(JSON.stringify(msg));
};


/**
 * Request to start the task on the server.
 */
var startServerTask = function (ws, time) {
  time = typeof time !== "undefined" ? time : 5;
  var msg = {type: "console", action: "start", time: time};
  ws.send(JSON.stringify(msg));
};


/**
 * Handle the received message of type "console"
 */
var handleMsgConsole = function (msg) {
  if (msg.action === "log") {
    // TODO: show the logging messages
  }
  else if (msg.action === "push") {
    // Update the task status
    updateTaskStatus(msg.status);
  }
  else if (msg.success) {
    setFormConfigs(msg.data, msg.errors);
  }
  else {
    console.error("WebSocket 'console' request failed:", msg.error);
    // TODO: add error code support and handle each specific error ...
  }
};
