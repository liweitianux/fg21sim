/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * WebSocket codes for the Web UI of "fg21sim"
 */


/**
 * Global variable
 * FIXME: try to avoid this ...
 */
var ws = null;  /* WebSocket */


/**
 * Get the full WebSocket URL from the given URI
 */
var getWebSocketURL = function (uri) {
  var proto = {"http:": "ws:", "https:": "wss:"}[location.protocol];
  var host = location.hostname;
  var port = location.port;
  var url = proto + "//" + host + ":" + port + uri;
  return url;
};


/**
 * Update the contents of element "#ws-status" according to the current
 * WebSocket status.
 */
var updateWSStatus = function (action) {
  if (action === "open") {
    // WebSocket opened and connected
    $("#ws-status").removeClass("label-default label-warning label-danger")
      .addClass("label-success");
    $("#ws-status .icon").removeClass("fa-question-circle fa-warning")
      .removeClass("fa-times-circle")
      .addClass("fa-check-circle");
    $("#ws-status .text").text("Connected");
  }
  else if (action === "close") {
    // WebSocket closed
    $("#ws-status").removeClass("label-default label-success label-danger")
      .addClass("label-warning");
    $("#ws-status .icon").removeClass("fa-question-circle fa-check-circle")
      .removeClass("fa-times-circle")
      .addClass("fa-warning");
    $("#ws-status .text").text("Disconnected!");
  }
  else if (action === "error") {
    // WebSocket encountered error
    $("#ws-status").removeClass("label-default label-success label-warning")
      .addClass("label-danger");
    $("#ws-status .icon").removeClass("fa-question-circle fa-check-circle")
      .removeClass("fa-warning")
      .addClass("fa-times-circle");
    $("#ws-status .text").text("Error!!");
  }
  else if (action === "unsupported") {
    // WebSocket NOT supported
    $("#ws-status").removeClass("label-default").addClass("label-danger");
    $("#ws-status .icon").removeClass("fa-question-circle")
      .addClass("fa-times-circle");
    $("#ws-status .text").text("Unsupported!!");
  }
  else {
    console.error("updateWSStatus: Unknown action: ", action);
  }
};


/**
 * Connect to WebSocket and bind functions to events
 */
var connectWebSocket = function (url) {
  ws = new WebSocket(url);
  ws.onopen = function () {
    console.log("Opened WebSocket: " + ws.url);
    updateWSStatus("open");
  };
  ws.onclose = function (e) {
    console.log("WebSocket closed because: ", e.reason);
    updateWSStatus("close");
    // Reconnect
    console.log("Reconnect the WebSocket in 1 second");
    setTimeout(function () { connectWebSocket(url); }, 1000);
  };
  ws.onerror = function (e) {
    console.error("WebSocket encountered error: ", e.message);
    updateWSStatus("error");
  };
  ws.onmessage = function (e) {
    console.log("WebSocket received message:");
    console.log(e.data);
  };
};


$(document).ready(function () {
  /**
   * Check `WebSocket` support
   */
  if (window.WebSocket) {
    // WebSocket supported
    console.log("Great, WebSocket is supported!");

    var ws_url = getWebSocketURL("/ws");
    connectWebSocket(ws_url);

  } else {
    // WebSocket NOT supported
    console.error("Oops, WebSocket is NOT supported!");
    updateWSStatus("unsupported");
  }
});
