/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * JavaScript codes for the Web UI of "fg21sim"
 */

"use strict";


/**
 * jQuery settings
 */
jQuery.ajaxSetup({
  error: function (error) {
    console.error("AJAX request failed: code:", error.status,
                  ", reason:", error.statusText); }
});


/**
 * Common functions that will be used by other scripts
 */

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
 * jQuery extension for easier AJAX JSON post
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
    success: function (response) {
      if (callback) {
        callback(response);
      }
    }
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
 *   - `message` : Main summary message
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
    $("<span>").addClass("fa fa-2x").addClass("fa-" + data.icon).appendTo(p);
  }
  if (data.message) {
    $("<span>").text(" " + data.message).appendTo(p);
  }
  modalBox.append(p);
  if (data.code) {
    modalBox.append($("<p>Error Code: </p>")
                    .append($("<span>")
                            .addClass("label label-warning")
                            .text(data.code)));
  }
  if (data.reason) {
    modalBox.append($("<p>Reason: </p>")
                    .append($("<span>")
                            .addClass("label label-warning")
                            .text(data.reason)));
  }
  if (data.buttons) {
    p = $("<p>").addClass("button-group");
    data.buttons.forEach(function (btn) {
      $("<button>").text(btn.text).addClass(btn["class"])
        .attr("type", "button")
        .on("click", btn.click).appendTo(p);
    });
  }
  modalBox.append(p);
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
});
