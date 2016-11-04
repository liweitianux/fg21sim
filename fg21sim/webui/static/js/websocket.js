/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * WebSocket codes for the Web UI of "fg21sim"
 */

"use strict";

/**
 * Global variable
 * FIXME: try to avoid this ...
 */
var ws = null;  /* WebSocket */
/* WebSocket reconnection settings */
var ws_reconnect = {
  maxTry: 100,
  tried: 0,
  timeout: 3000,  /* ms */
};


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
    console.error("updateWSStatus: Unknown action:", action);
  }
};


/**
 * Toggle the visibility of element "#ws-status".
 */
var toggleWSReconnect = function (action) {
  /**
   * Function default parameters: https://stackoverflow.com/a/894877/4856091
   */
  action = typeof action !== "undefined" ? action : "toggle";

  var target = $("#ws-reconnect");
  if (action === "toggle") {
    target.toggle();
  } else if (action === "show") {
    target.show();
  } else if (action === "hide") {
    target.hide();
  } else {
    console.error("toggleWSReconnect: Unknown action:", action);
  }
};


/**
 * Connect to WebSocket and bind functions to events
 */
var connectWebSocket = function (url) {
  ws = new WebSocket(url);
  ws.onopen = function () {
    console.log("Opened WebSocket:", ws.url);
    updateWSStatus("open");
    toggleWSReconnect("hide");
  };
  ws.onclose = function (e) {
    console.log("WebSocket closed: code:", e.code, ", reason:", e.reason);
    updateWSStatus("close");
    // Reconnect
    if (ws_reconnect.tried < ws_reconnect.maxTry) {
      ws_reconnect.tried++;
      console.log("Try reconnect the WebSocket: No." + ws_reconnect.tried);
      setTimeout(function () { connectWebSocket(url); },
                 ws_reconnect.timeout);
    } else {
      console.error("WebSocket already tried allowed maximum times:",
                    ws_reconnect.maxTry);
      toggleWSReconnect("show");
    }
  };
  ws.onerror = function (e) {
    console.error("WebSocket encountered error:", e.message);
    updateWSStatus("error");
    toggleWSReconnect("show");
  };
  ws.onmessage = function (e) {
    var msg = JSON.parse(e.data);
    console.log("WebSocket received message: type:", msg.type,
                ", status:", msg.status);
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

    // Bind event to the "#ws-reconnect" button
    $("#ws-reconnect").on("click", function () {
      console.log("WebSocket: reset the tried reconnection counter");
      ws_reconnect.tried = 0;
      console.log("Manually reconnect the WebSocket:", ws_url);
      connectWebSocket(ws_url);
    });

  } else {
    // WebSocket NOT supported
    console.error("Oops, WebSocket is NOT supported!");
    updateWSStatus("unsupported");
  }
});
