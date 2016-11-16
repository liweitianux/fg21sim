/**
 * Copyright (c) 2016 Weitian LI <liweitianux@live.com>
 * MIT license
 *
 * JavaScript codes for the Web UI of "fg21sim"
 */

"use strict";


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
