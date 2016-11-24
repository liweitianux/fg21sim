/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * JavaScript codes for the Web UI of "fg21sim"
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
 * jQuery AJAX global callbacks using the global AJAX event handler methods
 *
 * NOTE:
 * It is NOT recommended to use `jQuery.ajaxSetup` which will affect ALL calls
 * to `jQuery.ajax` or AJAX-based derivatives.
 */
$(document).ajaxError(function (event, jqxhr, settings, exception) {
  console.error("AJAX request failed: code:", jqxhr.status,
                ", reason:", jqxhr.statusText);
  if (jqxhr.status === 403) {
    // Forbidden error: redirect to login page
    window.location.href = "/login";
  }
});


/**
 * Extend jQuery with the `disable()` function to enable/disable buttons,
 * input, etc.
 *
 * Credit: https://stackoverflow.com/a/16788240/4856091
 */
jQuery.fn.extend({
  disable: function (state) {
    return this.each(function () {
      if ($(this).is("input, button, textarea, select")) {
        this.disabled = state;
      } else {
        $(this).toggleClass("disabled", state);
      }
    });
  }
});


/**
 * Get the value of a key stored in the cookie
 *
 * @param {String} name - Name of the key
 *
 * @return {String} - The value of the key; `undefined` if the key not exists
 *
 * Credit:
 * http://www.tornadoweb.org/en/stable/guide/security.html
 */
var getCookie = function (name) {
  var m = document.cookie.match("\\b" + name + "=([^;]*)\\b");
  return m ? m[1] : undefined;
};


/**
 * jQuery extension for handy AJAX POST request using JSON
 *
 * NOTE: The XSRF token is extracted from the cookie and posted together.
 *
 * @param {String} url - The URL that handles the AJAX requests
 * @param {Object} data - Data of key-value pairs to be posted
 * @param {Function} callback - Function to be called when AJAX succeeded
 */
jQuery.postJSON = function (url, data, callback) {
  return jQuery.ajax({
    url: url,
    type: "POST",
    contentType: "application/json; charset=utf-8",
    // Tornado: `check_xsrf_cookie()`
    // Credit: https://stackoverflow.com/a/28924601/4856091
    headers: {"X-XSRFToken": getCookie("_xsrf")},
    data: JSON.stringify(data),
    success: callback
  });
};


/**
 * jQuery extension for *uncached* AJAX GET request using JSON
 *
 * NOTE:
 * IE will by default cache the GET request even the contents has changed.
 *
 * Credit: https://stackoverflow.com/a/35130770/4856091
 *
 * @param {String} url - The URL that handles the AJAX requests
 * @param {Object} data - Data object sent to the server with the request
 * @param {Function} callback - Function to be called when AJAX succeeded
 */
jQuery.getJSONUncached = function (url, data, callback) {
  return jQuery.ajax({
    url: url,
    type: "GET",
    dataType: "json",
    data: data,
    // Force the requested page NOT to be cached by the browser!
    cache: false,
    success: callback
  });
};


/**
 * Scroll the page to adjust for the fixed navigation banner
 */
var scrollTarget = function (height) {
  if ($(":target").length) {
    var offset = $(":target").offset();
    var scroll_to = offset.top - height * 1.2;
    $("html, body").animate({scrollTop: scroll_to}, 100);
  }
};


/**
 * Toggle the display of the target block
 */
var toggleBlock = function (toggle, targetBlock) {
  if (targetBlock.is(":visible")) {
    targetBlock.slideUp("fast");
    toggle.removeClass("fa-chevron-circle-up")
      .addClass("fa-chevron-circle-down")
      .attr("title", "Expand contents");
  } else {
    targetBlock.slideDown("fast");
    toggle.removeClass("fa-chevron-circle-down")
      .addClass("fa-chevron-circle-up")
      .attr("title", "Collapse contents");
  }
};


/**
 * Compose the notification contents and shown them in the modal box.
 *
 * The input `modalBox` may be a jQuery object or a jQuery selector of the
 * target modal box.
 *
 * The input `data` may have the following attributes:
 *   - `icon` : FontAwesome icon (specified without the beginning `fa-`)
 *   - `title` : Notification title/summary
 *   - `contents` : Notification detail contents, may be a list of paragraphs
 *   - `code` : Error code if it is an error notification
 *   - `reason` : Reason of the error
 *   - `buttons` : A list of buttons, which have these attributes:
 *                 + `text` : Button name
 *                 + `class` : {String} Button classes
 *                 + `click` : {Function} Function called on click.
 *                             To close the modal, use `$.modal.close()`
 */
var showModal = function (modalBox, data) {
  modalBox = $(modalBox);
  // Empty previous contents
  modalBox.html("");
  var p = $("<p>");
  if (data.icon) {
    $("<span>").addClass("icon fa")
      .addClass("fa-" + data.icon).appendTo(p);
  }
  if (data.title) {
    $("<span>").addClass("title").text(data.title).appendTo(p);
  }
  modalBox.append(p);
  if (data.contents) {
    if ($.isArray(data.contents)) {
      data.contents.forEach(function (p) {
        modalBox.append($("<p class='contents'>").html(p));
      });
    } else {
      modalBox.append($("<p class='contents'>").html(data.contents));
    }
  }
  if (data.code) {
    modalBox.append($("<p>")
                    .append($("<span>").text("Code:")
                            .addClass("label label-warning"))
                    .append($("<span>").text(data.code)));
  }
  if (data.reason) {
    modalBox.append($("<p>")
                    .append($("<span>").text("Reason:")
                            .addClass("label label-warning"))
                    .append($("<span>").text(data.reason)));
  }
  if (data.buttons) {
    p = $("<p>").addClass("button-group");
    data.buttons.forEach(function (btn) {
      $("<button>").text(btn.text).addClass(btn["class"])
        .attr("type", "button")
        .on("click", btn.click).appendTo(p);
    });
    modalBox.append(p);
  }
  // Show the modal box
  modalBox.modal();
};


$(document).ready(function () {
  // Scroll the page to adjust for the fixed navigation banner
  $(window).on("hashchange", function () {
    var nav_height = $("nav.navigation").outerHeight();
    scrollTarget(nav_height);
  });

  // Toggle section contents/body
  $(".heading > .toggle").on("click", function () {
    var toggle = $(this);
    var body = toggle.closest(".heading").next(".body");
    toggleBlock(toggle, body);
  });

  // Panel toggle control
  $(".panel-title > .toggle").on("click", function () {
    var toggle = $(this);
    var body = toggle.closest(".panel").find(".panel-body");
    toggleBlock(toggle, body);
  });

  // Prevent from submitting form by "Enter"
  // Credit; https://stackoverflow.com/a/11235672/4856091
  $("form").on("keypress", function (e) {
    if (e.which === 13) {
      e.preventDefault();
      return false;
    }
  });
});
