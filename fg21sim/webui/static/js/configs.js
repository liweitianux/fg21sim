/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * Web UI of "fg21sim"
 * Configuration form manipulations using the WebSocket communications
 */

"use strict";


/**
 * Clear the error states previously marked on the fields with invalid values.
 */
var clearConfigFormErrors = function () {
  // TODO

  $("#conf-form").find(":input").each(function () {
    $(this);
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
 */
var loadServerConfigFile = function (ws) {
  var workdir = $("input[name=workdir]").val().replace(/[\\\/]$/, "");
  var configfile = $("input[name=configfile]").val().replace(/^.*[\\\/]/, "");
  // FIXME: should use the native system path separator!
  var filepath = workdir + "/" + configfile;
  var msg = {type: "configs", action: "load", userconfig: filepath};
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
  var workdir = $("input[name=workdir]").val().replace(/[\\\/]$/, "");
  var configfile = $("input[name=configfile]").val().replace(/^.*[\\\/]/, "");
  // FIXME: should use the native system path separator!
  var filepath = workdir + "/" + configfile;
  var msg = {type: "configs",
             action: "load",
             outfile: filepath,
             clobber: clobber};
  ws.send(JSON.stringify(msg));
};
