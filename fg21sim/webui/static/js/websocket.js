/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * WebSocket codes for the Web UI of "fg21sim"
 */

"use strict";

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
 * Toggle the visibility of element "#ws-status".
 */
var toggleWSReconnect = function (action) {
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
    toggleWSReconnect("hide");
  }
  else {
    console.error("updateWSStatus: Unknown action:", action);
  }
};


$(document).ready(function () {
  /**
   * Check "WebSocket" support
   */
  if (window.WebSocket) {
    // WebSocket supported
    console.log("Great, WebSocket is supported!");

    // Custom object for WebSocket handling
    var websocket = {
      // URL for the WebSocket connection
      url: getWebSocketURL("/ws"),

      // Allowed maximum number of reconnection times
      reconnectionMaxTry: 21,
      // Current number of tried reconnection times
      reconnectionTried: 0,
      // Wait time (unit: ms) before trying to reconnect
      reconnectionWaitTime: 3000,

      // Events handlers will be bound to the opened WebSocket object
      onopen: function () {
        console.log("Opened WebSocket:", this.url);
        updateWSStatus("open");
        toggleWSReconnect("hide");
      },
      onclose: function (e) {
        var self = this;
        console.log("WebSocket closed: code:", e.code, ", reason:", e.reason);
        updateWSStatus("close");
        // Try to reconnect
        if (self.reconnectionTried < self.reconnectionMaxTry) {
          self.reconnectionTried++;
          console.log("Try reconnect the WebSocket: No." +
                      self.reconnectionTried);
          setTimeout(function () { self.connect(); },
                     self.reconnectionWaitTime);
        } else {
          console.error("WebSocket already tried allowed maximum times:",
                        self.reconnectionMaxTry);
          toggleWSReconnect("show");
        }
      },
      onerror: function (e) {
        console.error("WebSocket encountered error:", e.message);
        updateWSStatus("error");
        toggleWSReconnect("show");
      },
      onmessage: function (e) {
        var msg = JSON.parse(e.data);
        console.log("WebSocket received message:", msg);
        // Delegate appropriate actions to handle the received message
        if (msg.type === "configs") {
          handleWebSocketMsgConfigs(msg);
        } else if (msg.type === "console") {
          handleWebSocketMsgConsole(msg);
        } else {
          // Unknown/unsupported message type
          console.warn("WebSocket: unknown message type:", msg.type);
        }
      },

      // Open the WebSocket and bind the events handlers
      connect: function () {
        var self = this;
        var ws = new WebSocket(self.url);
        ws.onopen = function (e) { self.onopen.call(self, e); };
        ws.onclose = function (e) { self.onclose.call(self, e); };
        ws.onerror = function (e) { self.onerror.call(self, e); };
        ws.onmessage = function (e) { self.onmessage.call(self, e); };
        this._websocket_ = ws;
      },

      // Force reconnect the WebSocket
      forceReconnect: function () {
        console.log("WebSocket: reset the tried reconnection counter");
        this.reconnectionTried = 0;
        console.log("Force reconnect the WebSocket:", this.url);
        this.connect();
      },

      // Close the WebSocket
      disconnect: function() {
        if (this._websocket_) {
          console.log("Disconnect the WebSocket:", this.url);
          this._websocket_.close();
          this._websocket_ = null;
        } else {
          console.warn("WebSocket already disconnected!");
        }
      },

      // Force disconnect the WebSocket
      forceDisconnect: function () {
        this.reconnectionTried = this.reconnectionMaxTry;
        console.log("Force disconnect the WebSocket:", this.url);
        this.disconnect();
      },

      // The opened WebSocket object
      _websocket_: null
    };

    // Add to the global "FG21SIM"
    FG21SIM.websocket = websocket;

    // Open the WebSocket connection
    websocket.connect();

    // Manually reconnect the WebSocket after tried allowed maximum times
    $("#ws-reconnect").on("click", function () {
      websocket.forceReconnect();
    });
  } else {
    // WebSocket NOT supported
    console.warn("Oops, WebSocket is NOT supported!");
    updateWSStatus("unsupported");
    showModal({
      icon: "warning",
      title: "WebSocket is NOT supported by the browser!",
      contents: ("The <strong>necessary functionalities</strong> do NOT " +
                 "depend on WebSocket. However, the user experience may be " +
                 "affected due to the missing WebSocket functionalities.")
    });
  }
});
