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
  var p = null;
  // Strip the trailing path separator
  path1 = path1.replace(/\/$/, "");
  if (path1 === "") {
    p = path2;
  } else {
    p = path1 + "/" + path2;
  }
  // Both "path1" and "path2" are empty
  if (p === "/") {
    console.error("Both 'path1' and 'path2' are empty");
    p = null;
  }
  return p;
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
 *
 * @returns value - value of the option field
 *                  + `null` if the field not exists or empty (no value)
 */
var getFormConfigSingle = function (name) {
  var value = null;
  if (! name) {
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
      } else if (target.is(":checkbox") && target.data("type") === "boolean") {
        // Convert the checkbox value into boolean
        value = target.prop("checked");
      } else if (target.is(":checkbox")) {
        // Get values of checked checkboxes into array
        // Credit: https://stackoverflow.com/a/16171146/4856091
        value = target.filter(":checked").map(
          function () { return $(this).val(); }).get();
      } else if (target.is(":text") && target.data("type") === "array") {
        // Convert back to Array
        value = target.val().split(/\s*,\s*/);
      } else {
        value = target.val();
      }
      // NOTE: convert "" (empty string) back to `null`
      if (value === "") {
        value = null;
      }
    } else {
      console.error("No such element:", selector);
    }
  }
  return value;
};


/**
 * Collect all the current configurations and values from the form.
 *
 * @returns {Object} key-value pairs of the form configurations
 */
var getFormConfigAll = function () {
  var names = $("#conf-form").find("input[name]").map(
    function () { return $(this).attr("name"); }).get();
  names = $.unique(names);
  var data = {};
  names.forEach(function (name) {
    data[name] = getFormConfigSingle(name);
  });
  // Do not forget the "userconfig"
  data["userconfig"] = getFormConfigSingle("userconfig");
  // Delete unwanted items
  ["workdir", "configfile", "_xsrf"].forEach(function (name) {
    delete data[name];
  });
  console.log("Collected form configurations data:", data);
  return data;
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
    } else {
      $("input[name=workdir]").val("");
      $("input[name=configfile]").val("");
    }
  } else {
    var selector = "input[name='" + name + "']";
    var target = $(selector);
    if (target.length) {
      if (target.is(":radio")) {
        target.val([value]);  // Use Array in "val()"
      } else if (target.is(":checkbox") && target.data("type") == "boolean") {
        // Convert the checkbox value into boolean
        target.prop("checked", value);
      } else if (target.is(":checkbox")) {
        // The received value is already an Array
        target.val(value);
      } else if (target.is(":text") && target.data("type") == "array") {
        // Convert array of values into a string
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
    if (val_old !== value) {
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
 * Update the configuration form status indicator: "#conf-status"
 *
 * NOTE:
 * Also store the current validity status in a custom data attribute:
 * `validity`, which has a boolean value.
 */
var updateFormConfigStatus = function () {
  var target = $("#conf-status");
  var recheck_icon = $("#conf-recheck");
  var invalid = $("#conf-form").find("input[name]:invalid");
  if (invalid.length) {
    // Exists invalid configurations
    console.warn("Found", invalid.length, "invalid configurations!");
    recheck_icon.show();
    target.removeClass("label-default label-success")
      .addClass("label-warning");
    target.find(".icon").removeClass("fa-question-circle fa-check-circle")
      .addClass("fa-warning");
    target.find(".text").text("Invalid!");
    target.data("validity", false);
  } else {
    // All valid
    // console.info("Great, all configurations are valid :)");
    recheck_icon.hide();
    target.removeClass("label-default label-warning")
      .addClass("label-success");
    target.find(".icon").removeClass("fa-question-circle fa-warning")
      .addClass("fa-check-circle");
    target.find(".text").text("OK");
    target.data("validity", true);
  }
};


/**
 * Show notification contents in the "#modal-configs" modal box.
 */
var showModalConfigs = function (data) {
  var modalBox = $("#modal-configs");
  showModal(modalBox, data);
};


/**
 * Get the configurations from the server and update the client form
 * to the newly received values.
 *
 * NOTE:
 * The configurations are not validated on the server, therefore,
 * there is no validation error returned.
 * For the validation, see function `validateServerConfigs()`.
 *
 * @param {String} url - The URL that handles the "configs" AJAX requests.
 * @param {Array} [keys=null] - List of keys whose values will be requested.
 *                              If `null` then request all configurations.
 */
var getServerConfigs = function (url, keys) {
  keys = typeof keys !== "undefined" ? keys : null;
  return $.getJSON(url, {action: "get", keys: JSON.stringify(keys)},
                   function (response) {
                     setFormConfigs(response.data, {});
                   });
};


/**
 * Validate the server-side configurations to get the validation errors,
 * and mark the corresponding form fields to be invalid with details.
 */
var validateServerConfigs = function (url) {
  return $.getJSON(url, {action: "validate"},
                   function (response) {
                     setFormConfigs({}, response.errors);
                   });
};


/**
 * Reset the server-side configurations to the defaults, then sync back to
 * the client-side form configurations.
 */
var resetConfigs = function (url) {
  $.postJSON(url, {action: "reset"})
    .done(function () {
      // Server-side configurations already reset
      resetFormConfigs();
      // Sync server-side configurations back to the client
      $.when(getServerConfigs(url),
             validateServerConfigs(url))
        .done(function () {
          // Update the configuration status label
          updateFormConfigStatus();
          // Popup a modal notification
          var modalData = {};
          modalData.icon = "check-circle";
          modalData.message = "Reset and synchronized the configurations.";
          showModalConfigs(modalData);
        });
    })
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.message = "Failed to reset the configurations!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalConfigs(modalData);
    });
};


/**
 * Set the server-side configurations using the sent data from the client.
 *
 * NOTE:
 * The supplied configuration data are validated on the server side, and
 * the validation errors are sent back.
 * However, the whole configurations is NOT checked, therefore, function
 * `validateServerConfigs()` should be used if necessary.
 *
 * @param {Object} [data={}] - Group of key-value pairs that to be sent to
 *                             the server to update the configurations there.
 */
var setServerConfigs = function (url, data) {
  data = typeof data !== "undefined" ? data : {};
  return $.postJSON(url, {action: "set", data: data},
                    function (response) {
                      setFormConfigs({}, response.errors);
                    })
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.message = "Failed to update/set the configuration data!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalConfigs(modalData);
    });
};


/**
 * Request the server to load/merge the configurations from the specified
 * user configuration file.
 *
 * @param {Object} userconfig - Absolute path to the user config file on the
 *                              server. If not specified, then determine from
 *                              the form fields "workdir" and "configfile".
 */
var loadServerConfigFile = function (url, userconfig) {
  if (! userconfig) {
    userconfig = getFormConfigSingle("userconfig");
  }
  return $.postJSON(url, {action: "load", userconfig: userconfig})
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.message = "Failed to load the user configuration file!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalConfigs(modalData);
    });
};


/**
 * Request the server to save current configurations to the supplied output
 * file.
 *
 * @param {Boolean} [clobber=false] - Whether overwrite the existing file.
 */
var saveServerConfigFile = function (url, clobber) {
  clobber = typeof clobber !== "undefined" ? clobber : false;
  var userconfig = getFormConfigSingle("userconfig");
  var data = {action: "save",
              outfile: userconfig,
              clobber: clobber};
  return $.postJSON(url, data)
    .done(function () {
      var modalData = {};
      if ($("#conf-status").data("validity")) {
        // Form configurations is valid :)
        modalData.icon = "check-circle";
        modalData.message = "Configurations saved to file.";
      } else {
        // Configurations is currently invalid!
        modalData.icon = "warning";
        modalData.message = ("Configurations saved to file. " +
                             "But there exist some invalid values!");
      }
      showModalConfigs(modalData);
    })
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.message = "Failed to save the configurations!";
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalConfigs(modalData);
    });
};


/**
 * Check whether the specified file already exists on the server?
 */
var existsServerFile = function (url, filepath, callback) {
  var data = {action: "exists",
              filepath: JSON.stringify(filepath)};
  return $.getJSON(url, data, callback)
    .fail(function (jqxhr) {
      var modalData = {};
      modalData.icon = "times-circle";
      modalData.message = ("Failed to check the existence " +
                           "of the user configuration file!");
      modalData.code = jqxhr.status;
      modalData.reason = jqxhr.statusText;
      showModalConfigs(modalData);
    });
};


/**
 * Handle the received message of type "configs" pushed through the WebSocket
 */
var handleWebSocketMsgConfigs = function (msg) {
  if (msg.action === "push") {
    // Pushed configurations (with validations) of current state on the server
    setFormConfigs(msg.data, msg.errors);
    updateFormConfigStatus();
  } else {
    console.warn("WebSocket: received message:", msg);
  }
};


$(document).ready(function () {
  // URL to handle the "configs" AJAX requests
  var ajax_url = "/ajax/configs";

  // Re-check/validate the whole form configurations
  $("#conf-recheck").on("click", function () {
    var data = getFormConfigAll();
    setServerConfigs(ajax_url, data)
      .then(function () { validateServerConfigs(ajax_url); })
      .done(function () { updateFormConfigStatus(); });
  });

  // Reset both server-side and client-side configurations to the defaults
  $("#reset-defaults").on("click", function () {
    var modalData = {};
    modalData.icon = "warning";
    modalData.message = ("Are you sure to reset the configurations?");
    modalData.buttons = [
      {
        text: "Cancel",
        click: function () { $.modal.close(); }
      },
      {
        text: "Reset!",
        "class": "button-warning",
        click: function () {
          $.modal.close();
          resetConfigs(ajax_url);
        }
      },
    ];
    showModalConfigs(modalData);
  });

  // Load the configurations from the specified user configuration file
  $("#load-configfile").on("click", function () {
    var userconfig = getFormConfigSingle("userconfig");
    resetFormConfigs();
    $.when(loadServerConfigFile(ajax_url, userconfig),
           getServerConfigs(ajax_url),
           validateServerConfigs(ajax_url))
      .done(function () {
        // Update the configuration status label
        updateFormConfigStatus();
        // Popup a modal notification
        var modalData = {};
        modalData.icon = "check-circle";
        modalData.message = "Loaded the configurations from file.";
        showModalConfigs(modalData);
      });
  });

  // Save the current configurations to file
  $("#save-configfile").on("click", function () {
    var userconfig = getFormConfigSingle("userconfig");
    existsServerFile(ajax_url, userconfig, function (response) {
      if (response.data.exists) {
        // The specified configuration file already exists
        // Confirm to overwrite
        var modalData = {};
        modalData.icon = "warning";
        modalData.message = ("Configuration file already exists! Overwrite?");
        modalData.buttons = [
          {
            text: "Cancel",
            rel: "modal:close",
            click: function () { $.modal.close(); }
          },
          {
            text: "Overwrite!",
            "class": "button-warning",
            rel: "modal:close",
            click: function () {
              $.modal.close();
              saveServerConfigFile(ajax_url, true);
            }
          },
        ];
        showModalConfigs(modalData);
      } else {
        saveServerConfigFile(ajax_url, false);
      }
    });
  });

  // Sync changed field to server, validate and update form
  $("#conf-form input").on("change", function (e) {
    console.log("Element changed:", e);
    var name = $(e.target).attr("name");
    var value = getFormConfigSingle(name);
    // Synchronize the changed form configuration to the server
    // NOTE:
    // Use the "computed property names" available in ECMAScript 6
    // (IE 11 not support this!)
    // var data = {[name]: value};
    var data = {};
    data[name] = value;
    setServerConfigs(ajax_url, data)
      .then(function () { return validateServerConfigs(ajax_url); })
      .done(function () { updateFormConfigStatus(); });
  });

  // When field "common/nside" changed, update the resolution note, as well
  // as the maximum multiple "common/lmax"
  $("#conf-form input[name='common/nside']").on("change", function (e) {
    var nside = parseInt($(this).val());
    // Update the resolution note (unit: arcmin)
    var resolution = Math.sqrt(3/Math.PI) * 3600 / nside;
    $(this).closest(".form-group").find(".note > .value")
      .text(resolution.toFixed(2));
    // Also update the maximum multipole "common/lmax"
    if (! isNaN(nside)) {
      var lmax = 3 * nside - 1;
      $("#conf-form input[name='common/lmax']").val(lmax).trigger("change");
    }
  });
  $("#conf-form input[name='common/nside']").keypress(function (e) {
    if (e.which === 13) {
      var nside = parseInt($(this).val());
      // Update the resolution note (unit: arcmin)
      var resolution = Math.sqrt(3/Math.PI) * 3600 / nside;
      $(this).closest(".form-group").find(".note > .value")
        .text(resolution.toFixed(2));
    }
  });
});
