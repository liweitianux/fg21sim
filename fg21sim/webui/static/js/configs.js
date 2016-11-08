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
 * Set custom error messages for the form fields that failed the server-side
 * validations.
 *
 * NOTE:
 * Add "error" class for easier manipulations, e.g., clearFormConfigErrors().
 *
 * References: Constraint Validation API
 *
 * @param {String} name - The name of filed name
 * @param {String} error - The custom error message to be set for the field
 */
var setFormConfigErrorSingle = function (name, error) {
  var selector = null;
  if (name === "userconfig") {
    selector = "input[name=configfile]";
  } else {
    selector = "input[name='" + name + "']";
  }
  $(selector).each(function () {
    // Reset the error message
    this.setCustomValidity(error);
    // Also add the "error" class for easier use later
    $(this).addClass("error");
  });
};


/**
 * Clear the custom error states marked on the form fields that failed the
 * server-side validations.
 *
 * NOTE: The form fields marked custom errors has the "error" class.
 *
 * References: Constraint Validation API
 */
var clearFormConfigErrors = function () {
  $("input.error").each(function () {
    // Remove the dynamically added "error" class
    $(this).removeClass("error");
    // Reset the error message
    this.setCustomValidity("");
  });
};


/**
 * Reset the configuration form to its defaults as written in the HTML.
 */
var resetFormConfigs = function () {
  // Credit: http://stackoverflow.com/a/6364313
  $("#conf-form")[0].reset();

  // Clear previously marked errors
  clearFormConfigErrors();
};


/**
 * Get the value of one single form field by specifying the name.
 *
 * @param {String} name - The name of filed name
 */
var getFormConfigSingle = function (name) {
  var value = null;
  if (name == null) {
    // do nothing
  } else if (name === "userconfig") {
    value = joinPath($("input[name=workdir]").val(),
                     $("input[name=configfile]").val());
  } else {
    var selector = "input[name='" + name + "']";
    var target = $(selector);
    if (target.length) {
      if (target.is(":radio")) {
        value = target.filter(":checked").val();
      } else if (target.is(":checkbox")) {
        // Get values of checked checkboxes into array
        // Credit: https://stackoverflow.com/a/16171146/4856091
        value = target.filter(":checked").map(
          function () { return $(this).val(); }).get();
      } else {
        value = target.val();
      }
    } else {
      console.error("No such element:", selector);
    }
  }
  return value;
};


/**
 * Set the value of one single form field according to the given
 * name and value.
 *
 * NOTE: Do NOT manually trigger the "change" event.
 *
 * @param {String} name - The name of filed name
 * @param {String|Number|Array} value - The value to be set for the field
 */
var setFormConfigSingle = function (name, value) {
  if (name === "userconfig") {
    if (value) {
      // Split the absolute path to "workdir" and "configfile"
      var workdir = dirname(value);
      var configfile = basename(value);
      $("input[name=workdir]").val(workdir);
      $("input[name=configfile]").val(configfile);
    }
  } else {
    var selector = "input[name='" + name + "']";
    var target = $(selector);
    if (target.length) {
      if (target.is(":radio")) {
        target.val([value]);  // Use Array in "val()"
      } else if (target.is(":checkbox")) {
        // Convert value (key of a single option) to an Array
        if (! Array.isArray(value)) {
          value = [value];
        }
        target.val(value);
      } else if (target.is(":text") && target.data("type") == "array") {
        // The received value is already an Array
        value = value.join(", ");
        target.val(value);
      } else {
        target.val(value);
      }
    } else {
      console.error("No such element:", selector);
    }
  }
};


/**
 * Set the configuration form to the supplied data, and mark out the fields
 * with error states as specified in the given errors.
 *
 * @param {Object} data - The input configurations data, key-value pairs.
 * @param {Object} errors - The config options with invalid values.
 */
var setFormConfigs = function (data, errors) {
  // Set the values of form field to the input configurations data
  $.each(data, function (name, value) {
    if (value == null) {
      value = "";  // Default to empty string
    }
    var val_old = getFormConfigSingle(name);
    if (val_old != null && val_old !== value) {
      setFormConfigSingle(name, value);
      console.log("Set input '" + name + "' to:", value, " <-", val_old);
    }
  });

  // Clear previously marked errors
  clearFormConfigErrors();

  // Mark custom errors on fields with invalid values validated by the server
  $.each(errors, function (name, error) {
    setFormConfigErrorSingle(name, error);
  });
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
    userconfig = getFormConfigSingle("userconfig");
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
  var userconfig = getFormConfigSingle("userconfig");
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
    setFormConfigs(msg.data, msg.errors);
  } else {
    console.error("WebSocket 'configs' request failed with error:", msg.error);
    // TODO: add error code support and handle each specific error ...
  }
};
