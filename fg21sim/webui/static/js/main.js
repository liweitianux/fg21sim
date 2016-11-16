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


$(document).ready(function () {
  // Scroll the page to adjust for the fixed navigation banner
  $(window).on("hashchange", function () {
    var nav_height = $("nav.navigation").outerHeight();
    scrollTarget(nav_height);
  });
});
