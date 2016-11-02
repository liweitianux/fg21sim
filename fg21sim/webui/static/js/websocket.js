/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * WebSocket codes for the Web UI of "fg21sim"
 */


$(document).ready(function () {
  /**
   * Check `WebSocket` support
   */
  if (window.WebSocket) {
    // WebSocket supported
    var createWebSocket = function (uri) {
      $("#ws-status .text").text("Supported");
      var proto = {"http:": "ws:", "https:": "wss:"}[location.protocol];
      var host = location.hostname;
      var port = location.port;
      var url = proto + "//" + host + ":" + port + uri;
      var socket = new WebSocket(url);
      return socket;
    };

    var ws = createWebSocket("/ws");
    ws.onopen = function () {
      $("#ws-status").removeClass("label-default label-danger")
                     .addClass("label-success");
      $("#ws-status .icon").removeClass("fa-question-circle fa-warning")
                           .addClass("fa-check-circle");
      $("#ws-status .text").text("Connected");
    };
    ws.onclose = function () {
      $("#ws-status").removeClass("label-default label-success")
                     .addClass("label-danger");
      $("#ws-status .icon").removeClass("fa-question-circle fa-check-circle")
                           .addClass("fa-warning");
      $("#ws-status .text").text("Disconnected!");
    };
    ws.onmessage = function (e) {
      console.log("WS received: " + e.data);
    };

  } else {
    // WebSocket NOT supported
    $("#ws-status").removeClass("label-default").addClass("label-danger");
    $("#ws-status .icon").removeClass("fa-question-circle")
                         .addClass("fa-times-circle");
    $("#ws-status .text").text("Unsupported!!");
  }
});
