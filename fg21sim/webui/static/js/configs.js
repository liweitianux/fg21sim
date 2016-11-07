/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * Web UI of "fg21sim"
 * Configuration form manipulations using the WebSocket communications
 */

"use strict";


/**
 * Generic utilities
 */

/**
 * Get the basename of a path
 * FIXME: only support "/" as the path separator
 */
var basename = function (path) {
  return path.replace(/^.*\//, "");
};

/**
 * Get the dirname of a path
 * FIXME: only support "/" as the path separator
 */
var dirname = function (path) {
  var dir = path.replace(/\/[^\/]*\/?$/, "");
  if (dir === "") {
    dir = "/";
  }
  return dir;
};

/**
 * Join the two path
 * FIXME: only support "/" as the path separator
 */
var joinPath = function (path1, path2) {
  // Strip the trailing path separator
  path1 = path1.replace(/\/$/, "");
  return (path1 + "/" + path2);
};



/**
 * Clear the error states previously marked on the fields with invalid values.
 */
var clearConfigFormErrors = function () {
  // TODO
  $("#conf-form").find(":input").each(function () {
    // TODO
  });
};


/**
 * Reset the form to its defaults as written in the HTML.
 */
var resetConfigForm = function () {
  // Credit: http://stackoverflow.com/a/6364313
  $("#conf-form")[0].reset();

  // TODO: whether this is required ??
  // Clear previously marked errors
  clearConfigFormErrors();
};


/**
 * Set the configuration form to the supplied data, and mark out the fields
 * with error states as specified in the given errors.
 *
 * @param {Object} data - The input configurations data, key-value pairs.
 * @param {Object} errors - The config options with invalid values.
 */
var setConfigForm = function (data, errors) {
  // Clear previously marked errors
  clearConfigFormErrors();

  // Set the values of form field to the input configurations data
  for (var key in data) {
    if (! data.hasOwnProperty(key)) {
      /**
       * NOTE: Skip if the property is from prototype
       * Credit: http://stackoverflow.com/a/921808
       */
      continue;
    }
    var value = data[key];
    if (key === "userconfig" && value) {
      // Split the absolute path to "workdir" and "configfile"
      var workdir = dirname(value);
      var configfile = basename(value);
      $("input[name=workdir]").val(workdir).trigger("change");
      $("input[name=configfile]").val(configfile).trigger("change");
    }
    else {
      var selector = "input[name='" + key + "']";
      var target = $(selector);
      if (target.length) {
        if (target.is(":radio")) {
          var val_old = target.filter(":checked").val();
          target.val([value]).trigger("change");  // Use Array in "val()"
        } else if (target.is(":checkbox")) {
          // Get values of checked checkboxes into array
          // Credit: https://stackoverflow.com/a/16171146/4856091
          var val_old = target.filter(":checked").map(
            function () { return $(this).val(); }).get();
          // Convert value to an Array
          if (! Array.isArray(value)) {
            value = [value];
          }
          target.val(value).trigger("change");
        } else if (target.is(":text") && target.data("type") == "array") {
          // This field is a string that is ", "-joined from an Array
          var val_old = target.val();
          // The received value is already an Array
          value = value.join(", ");
          target.val(value).trigger("change");
        } else {
          var val_old = target.val();
          target.val(value).trigger("change");
        }
        console.debug("Set input '" + key + "' to:", value, " <-", val_old);
      }
      else {
        console.error("No such element:", selector);
      }
    }
  }

  // Mark error states on fields with invalid values
  for (var key in errors) {
    if (! errors.hasOwnProperty(key)) {
      // NOTE: Skip if the property is from prototype
      continue;
    }
    var value = errors[key];
    // TODO: mark the error states
  }
};


/**
 * Get the filepath to the user configuration file from the form fields
 * "workdir" and "configfile".
 *
 * @returns {String} - Absolute path to the user configuration file.
 */
var getFormUserconfig = function () {
  var userconfig = joinPath($("input[name=workdir]").val(),
                            $("input[name=configfile]").val());
  return userconfig;
};


/**
 * Reset the server-side configurations to the defaults.
 *
 * @param {Object} ws - The opened WebSocket object, through which send
 *                      the request for configuration defaults.
 */
var resetServerConfigs = function (ws) {
  var msg = {type: "configs", action: "reset"};
  ws.send(JSON.stringify(msg));
};


/**
 * Get the configurations from the server.
 * When the response arrived, the bound function will take appropriate
 * reactions (e.g., `setConfigForm()`) to update the form contents.
 *
 * @param {Array} [keys=null] - List of keys whose values will be requested.
 *                              If `null` then request all configurations.
 *
 */
var getServerConfigs = function (ws, keys) {
  keys = typeof keys !== "undefined" ? keys : null;
  var msg = {type: "configs", action: "get", keys: keys};
  ws.send(JSON.stringify(msg));
};


/**
 * Set the server-side configurations using the sent data from the client.
 *
 * NOTE: The server will validate the values and further check the whole
 *       configurations, and response the config options with invalid values.
 *
 * @param {Object} [data={}] - Group of key-value pairs that to be sent to
 *                             the server to update the configurations there.
 */
var setServerConfigs = function (ws, data) {
  data = typeof data !== "undefined" ? data : {};
  var msg = {type: "configs", action: "set", data: data};
  ws.send(JSON.stringify(msg));
};


/**
 * Request the server side configurations with user configuration file merged.
 * When the response arrived, the bound function will delegate an appropriate
 * function (i.e., `setConfigForm()`) to update the form contents.
 *
 * @param {Object} userconfig - Absolute path to the user config file on the
 *                              server. If not specified, then determine from
 *                              the form fields "workdir" and "configfile".
 */
var loadServerConfigFile = function (ws, userconfig) {
  if (typeof userconfig === "undefined") {
    userconfig = getFormUserconfig();
  }
  var msg = {type: "configs", action: "load", userconfig: userconfig};
  ws.send(JSON.stringify(msg));
};


/**
 * Request the server side configurations with user configuration file merged.
 * When the response arrived, the bound function will delegate an appropriate
 * function (i.e., `setConfigForm()`) to update the form contents.
 *
 * @param {Boolean} [clobber=false] - Whether overwrite the existing file.
 */
var saveServerConfigFile = function (ws, clobber) {
  clobber = typeof clobber !== "undefined" ? clobber : false;
  var userconfig = getFormUserconfig();
  var msg = {type: "configs",
             action: "save",
             outfile: userconfig,
             clobber: clobber};
  ws.send(JSON.stringify(msg));
};


/**
 * Handle the received message of type "configs".replace
 */
var handleMsgConfigs = function (msg) {
  if (msg.success) {
    setConfigForm(msg.data, msg.errors);
  } else {
    console.error("WebSocket 'configs' request failed with error:", msg.error);
    // TODO: add error code support and handle each specific error ...
  }
};
